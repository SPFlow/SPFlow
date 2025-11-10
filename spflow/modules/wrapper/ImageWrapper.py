from typing import Optional, Union

import torch
from torch import Tensor

from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.modules.module import Module
from spflow.modules.wrapper.abstract_wrapper import AbstractWrapper


class MarginalizationContext:
    def __init__(self, c: list[int] = None, h: list[int] = None, w: list[int] = None):
        self.c = [] if c is None else c
        self.h = [] if h is None else h
        self.w = [] if w is None else w


class ImageWrapper(AbstractWrapper):
    def __init__(self, module: Module, num_channel: int, height: int, width: int):
        """
        Initialize an ImageWrapper.

        Args:
            module: Module to wrap.
            num_channel: Number of channels in the image.
            height: Height of the image.
            width: Width of the image.
        """
        super().__init__(module=module)
        self.num_channel = num_channel
        self.height = height
        self.width = width
        assert len(
            module.scope.query) == height * width * num_channel, f"Module out_features {module.out_features} does not match the expected size {height * width * num_channel}."

    def flatten(self, tensor: torch.Tensor):
        assert tensor.dim() == 4, f"Input tensor must be 4-dimensional but got {tensor.dim()}-dimensional."
        assert tensor.shape[
                   1] == self.num_channel, f"Input tensor channel dimension must; match num_channel {self.num_channel} but got {tensor.shape[1]}."

        return tensor.view(tensor.shape[0], -1)

    def to_image_format(self, tensor: torch.Tensor, batch: bool = True):

        if batch:
            assert tensor.dim() == 2, f"Input tensor must be 2-dimensional but got {tensor.dim()}-dimensional."
            return tensor.view(tensor.shape[0], self.num_channel, self.height, self.width)
        else:
            assert tensor.dim() == 1, f"Input tensor must be 1-dimensional but got {tensor.dim()}-dimensional."
            return tensor.view(self.num_channel, self.height, self.width)

    def extra_repr(self):
        return f"C={self.num_channel}, H={self.height}, W={self.width}"


@dispatch(memoize=True)  # type: ignore
def em(
        wrapper: ImageWrapper,
        data: Tensor,
        check_support: bool = True,
        dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for the given leaf module.

    Args:
        leaf:
            Leaf module to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    assert data.shape == (data.shape[0], wrapper.num_channel, wrapper.height,
                          wrapper.width), f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."

    data = wrapper.flatten(data)
    em(wrapper.module, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
        wrapper: ImageWrapper,
        data: Tensor,
        check_support: bool = True,
        dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    r"""Computes log-likelihoods for the leaf module given the data.

    Missing values (i.e., NaN) are marginalized over.

    Args:
        leaf:
            Leaf to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the distribution.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.

    Raises:
        ValueError: Data outside of support.
    """

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    assert data.shape == (data.shape[0], wrapper.num_channel, wrapper.height,
                          wrapper.width), f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."
    data = wrapper.flatten(data)

    log_prob = log_likelihood(wrapper.module, data, check_support, dispatch_ctx)

    return log_prob


@dispatch  # type: ignore
def sample(
        wrapper: ImageWrapper,
        data: Tensor,
        is_mpe: bool = False,
        check_support: bool = True,
        dispatch_ctx: Optional[DispatchContext] = None,
        sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples specified numbers of instances from modules in the ``base`` backend without any evidence.

    Samples a specified number of instance from the module by creating an empty two-dimensional NumPy array (i.e., filled with NaN values) of appropriate size and filling it.

    Args:
        module:
            Module to sample from.
        num_samples:
            Number of samples to generate.
        is_mpe:
            Boolean value indicating whether to perform maximum a posteriori estimation (MPE).
            Defaults to False.
        check_support:
            Boolean value indicating whether if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional NumPy array containing the sampled values.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    sample(
        wrapper.module,
        data,
        is_mpe=is_mpe,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )
    return wrapper.to_image_format(data)
    # return data.view(data.shape[0], wrapper.num_channel, wrapper.height, wrapper.width)


@dispatch(memoize=True)  # type: ignore
def marginalize(
        wrapper: ImageWrapper,
        marg_ctx: MarginalizationContext,
        prune: bool = True,
        dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, ImageWrapper]:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    marg_rvs = []
    scope = torch.tensor(wrapper.scope.query)

    scope = wrapper.to_image_format(scope, batch=False)

    reduction_h = len(marg_ctx.h)
    reduction_w = len(marg_ctx.w)
    reduction_c = len(marg_ctx.c)
    for height_id in marg_ctx.h:
        marg_rvs.extend(scope[:, height_id, :].flatten().tolist())
    for width_id in marg_ctx.w:
        marg_rvs.extend(scope[:, :, width_id].flatten().tolist())
    for channel_id in marg_ctx.c:
        marg_rvs.extend(scope[channel_id, :, :].flatten().tolist())

    marg_input = marginalize(wrapper.module, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

    if marg_input is None:
        return None

    else:
        return ImageWrapper(module=marg_input, height=wrapper.height - reduction_h, width=wrapper.width - reduction_w,
                            num_channel=wrapper.num_channel - reduction_c)
