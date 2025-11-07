from typing import Optional, Union, Dict
from collections.abc import Callable

from spflow.modules.wrapper.abstract_wrapper import AbstractWrapper
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.modules.module import Module
from spflow.meta.data import Scope
from torch import Tensor
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.utils.leaf import apply_nan_strategy
import torch
class Pixel():
    def __init__(self,channel_idx: int, h: int, w: int):
        self.channel_idx = channel_idx
        self.h = h
        self.w = w

class MarginalizeContext():
    def __init__(self,channel_idx: list[int] = None, h: list[int] = None, w: list[int] = None):
        self.channel_idx = [] if channel_idx is None else channel_idx
        self.h = [] if h is None else h
        self.w = [] if w is None else w

class ImageWrapper(AbstractWrapper):
    def __init__(self, module: Module, num_channel: int, height: int, width: int, implicit_product: bool = True):
        """
        Initialize an ImageWrapper.

        Args:
            module: Module to wrap.
            num_channel: Number of channels in the image.
            height: Height of the image.
            width: Width of the image.
            implicit_product: Whether to use implicit product representation.
        """
        super().__init__(module=module)
        self._num_channel = num_channel
        self._height = height
        self._width = width
        self.implicit_product = implicit_product
        assert len(module.scope.query) == height * width * num_channel, f"Module out_features {module.out_features} does not match the expected size {height * width * num_channel}."


    @property
    def out_features(self) -> int:
        return self.module.out_features

    @property
    def num_channel(self) -> int:
        return self._num_channel

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def feature_to_scope(self) -> list[Scope]:
        return self.inputs.feature_to_scope

    @property
    def out_channels(self) -> int:
        return self.module.out_channels

    def pixel_to_scope(self, p: Pixel) -> Scope:
        """Convert a pixel (channel, height, width) to a scope index.

        Args:
            p:
                Pixel to convert.

        Returns:
            Corresponding scope index.
        """
        assert 0 <= p.channel_idx < self.num_channel, f"Channel index must be in [0, {self.num_channel}) but got {p.channel_idx}."
        assert 0 <= p.h < self.out_features, f"Height index must be in [0, {self.out_features}) but got {p.h}."
        assert 0 <= p.w < self.out_features, f"Width index must be in [0, {self.out_features}) but got {p.w}."

        scope = p.channel_idx * self.out_features + p.h * (self.out_features // 2) + p.w

        scope = Scope([scope])

        return scope

    def pixel_to_index(self, p: Pixel) -> int:
        """Convert a pixel (channel, height, width) to a scope index.

        Args:
            p:
                Pixel to convert.

        Returns:
            Corresponding scope index.
        """
        assert 0 <= p.channel_idx < self.num_channel, f"Channel index must be in [0, {self.num_channel}) but got {p.channel_idx}."
        assert 0 <= p.h < self.out_features, f"Height index must be in [0, {self.out_features}) but got {p.h}."
        assert 0 <= p.w < self.out_features, f"Width index must be in [0, {self.out_features}) but got {p.w}."

        idx = p.channel_idx * self.out_features + p.h * (self.out_features // 2) + p.w

        return idx

    def scope_to_pixel(self, scope: Scope) -> Pixel:
        """Convert a scope index to a pixel (channel, height, width).

        Args:
            scope:
                Scope to convert.

        Returns:
            Corresponding pixel.
        """

        idx = scope.query[0]

        c = idx // (self.out_features)
        h = (idx % (self.out_features)) // (self.out_features // 2)
        w = (idx % (self.out_features)) % (self.out_features // 2)

        p = Pixel(channel_idx=c, h=h, w=w)

        return p


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

    assert data.shape == (data.shape[0], wrapper.num_channel, wrapper.height, wrapper.width), f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."

    # Reshape the data to (batch_size, num_channel * num_features, ...)

    data = data.view(data.shape[0], -1)

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

    assert data.shape == (data.shape[0], wrapper.num_channel, wrapper.height, wrapper.width), f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."
    data = data.view(data.shape[0], -1)

    log_prob = log_likelihood(wrapper.module, data, check_support, dispatch_ctx)

    ## reshape log-likelihood to (batch_size, num_channel, out_features, num_repetitions)
    #if wrapper.num_repetitions is None:
    #    log_prob = log_prob.view(data.shape[0], wrapper.num_channel, wrapper.height, wrapper.width, wrapper.out_channels)
    #else:
    #    log_prob = log_prob.view(data.shape[0], wrapper.num_channel, wrapper.height, wrapper.width, wrapper.out_channels, wrapper.num_repetitions)
#
#    # if implicit product, sum over channel dimension
#    if wrapper.implicit_product:
 #       log_prob = log_prob.sum(dim=1)

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
    return data.view(data.shape[0], wrapper.num_channel, wrapper.height, wrapper.width)

@dispatch(memoize=True)  # type: ignore
def marginalize(
    wrapper: ImageWrapper,
    #marg_rvs: Union[list[Pixel], list[int]],
    marg_ctx: MarginalizeContext,
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, ImageWrapper]:

    # ToDo: only full column or full rows or full channel...

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    #if isinstance(marg_rvs[0], Pixel):
    #    marg_rvs = [wrapper.pixel_to_index(pixel) for pixel in marg_rvs]
    #p.channel_idx * self.out_features + p.h * (self.out_features // 2) + p.w
    marg_rvs = []
    scope = torch.tensor(wrapper.scope.query)
    scope = scope.view(wrapper.num_channel, wrapper.height, wrapper.width)

    red_height = len(marg_ctx.h)
    red_width = len(marg_ctx.w)
    red_channel = len(marg_ctx.channel_idx)
    for height_id in marg_ctx.h:
        marg_rvs.extend(scope[:, height_id, :].flatten().tolist())
    for width_id in marg_ctx.w:
        marg_rvs.extend(scope[:, :, width_id].flatten().tolist())
    for channel_id in marg_ctx.channel_idx:
        marg_rvs.extend(scope[channel_id, :, :].flatten().tolist())




    marg_input = marginalize(wrapper.module, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)


    if marg_input is None:
        return None

    else:
        return ImageWrapper(module=marg_input, height=wrapper.height - red_height, width=wrapper.width - red_width, num_channel=wrapper.num_channel - red_channel, implicit_product=wrapper.implicit_product)
