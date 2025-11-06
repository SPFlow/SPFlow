from abc import ABC
from typing import Optional, Union
from collections.abc import Callable
from abc import abstractmethod

import torch
from torch import Tensor

from spflow.distributions.distribution import Distribution
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import apply_nan_strategy
from spflow.utils.leaf import parse_leaf_args


class CondLeafModule(LeafModule):
    def __init__(self, scope: Scope, out_channels: int = None,
                 cond_f: Optional[Union[Callable, list[Callable]]] = None):
        """
        Initialize a Normal distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensors.
            mean (Tensor, optional): The mean parameter tensor.
            std (Tensor, optional): The standard deviation parameter tensor.
        """

        #mean, std = self.retrieve_params(data=torch.tensor([]), dispatch_ctx=init_default_dispatch_context())
        #event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[mean, std])
        if out_channels is None:
            raise ValueError("out_channels must be provided")
        super().__init__(scope, out_channels=out_channels)
        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Union[list[Callable], Callable]] = None) -> None:

        if isinstance(cond_f, list) and len(cond_f) != self.out_channels:
            raise ValueError(
                "'CondLeafModule' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    @abstractmethod
    def retrieve_params(
            self, data: torch.Tensor, dispatch_ctx: DispatchContext
    ):
        pass

    @abstractmethod
    def set_params(self, *params):
        pass



@dispatch(memoize=True)  # type: ignore
def em(
    leaf: LeafModule,
    data: torch.Tensor,
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
    # TODO: resolve this circular import somehow
    from spflow import maximum_likelihood_estimation

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = dispatch_ctx.cache["log_likelihood"][leaf].grad
        # normalize expectations for better numerical stability
        # Reduce expectations to shape [batch_size, 1]
        dims = list(range(1, len(expectations.shape)))
        expectations = expectations.sum(dims)
        expectations /= expectations.sum(dim=None, keepdim=True)

        # ----- maximization step -----

        # update parameters through maximum weighted likelihood estimation
        maximum_likelihood_estimation(
            leaf,
            data,
            weights=expectations,
            bias_correction=False,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )

    # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: CondLeafModule,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    r"""Computes log-likelihoods for the leaf module given the data.

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
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

    *params, = leaf.retrieve_params(torch.tensor([]), dispatch_ctx)
    leaf.set_params(*params)


    # get information relevant for the scope
    data = data[:, leaf.scope.query]

    # ----- marginalization -----
    marg_mask = torch.isnan(data)

    # If there are any marg_ids, set them to 0.0 to ensure that distribution.log_prob call is succesfull and doesn't throw errors
    # due to NaNs
    if marg_mask.any():
        data[marg_mask] = 0.0  # ToDo in-support value

    # ----- log probabilities -----

    # Unsqueeze scope_data to make space for num_nodes dimension
    data = data.unsqueeze(2)

    if check_support:
        # create mask based on distribution's support
        valid_mask = leaf.distribution.check_support(data)

        if not torch.all(valid_mask):
            raise ValueError(f"Encountered data instances that are not in the support of the distribution.")

    # compute probabilities for values inside distribution support
    log_prob = leaf.distribution.log_prob(data.float())

    # Marginalize entries
    log_prob[marg_mask] = 0.0

    # Set marginalized scope data back to NaNs
    if marg_mask.any():
        data[marg_mask] = torch.nan

    return log_prob


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: CondLeafModule,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of a leaf module.

    Weights are normalized to sum up to :math:`N`.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one-dimensional PyTorch tensor containing non-negative weights for all data samples.
            Must match number of samples in ``data``.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another PyTorch tensor of same size.
            Defaults to None.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        ValueError: Invalid arguments.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    *params, = leaf.retrieve_params(torch.tensor([]), dispatch_ctx)
    leaf.set_params(*params)

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    # apply NaN strategy
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # Forward to the actual distribution
    leaf.distribution.maximum_likelihood_estimation(scope_data, weights, bias_correction)


@dispatch  # type: ignore
def sample(
    module: CondLeafModule,
    data: Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from the leaf nodes in the ``torch`` backend given potential evidence.

    Samples missing values proportionally to its probability distribution function (PDF).

    Args:
        module:
            Leaf module to sample from.
        data:
            Two-dimensional PyTorch tensor containing potential evidence.
            Each row corresponds to a sample.
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
        Two-dimensional PyTorch tensor containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    *params, = module.retrieve_params(torch.tensor([]), dispatch_ctx)
    module.set_params(*params)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    out_of_scope = list(filter(lambda x: x not in module.scope.query, range(data.shape[1])))
    marg_mask = torch.isnan(data)
    marg_mask[:, out_of_scope] = False

    # Mask that tells us which feature at which sample is relevant and should be sampled
    samples_mask = marg_mask
    samples_mask[:, module.scope.query] &= sampling_ctx.mask
    # samples_mask = sampling_ctx.mask & marg_mask

    # Count number of samples to draw
    instance_mask = samples_mask.sum(1) > 0
    n_samples = instance_mask.sum()  # count number of rows which have at least one true value

    if is_mpe:
        # Get mode of distribution as MPE
        samples = module.distribution.mode()

        # Add batch dimension
        samples = samples.unsqueeze(0).repeat(n_samples, *([1] * (samples.dim())))
    else:
        # Sample from distribution
        samples = module.distribution.sample(n_samples=n_samples)

    if samples.shape[0] != sampling_ctx.channel_index[instance_mask].shape[0]:
        raise ValueError(f"Sample shape mismatch: got {samples.shape[0]}, expected {sampling_ctx.channel_index[instance_mask].shape[0]}")

    if module.out_channels == 1:
        # If the output of the input module has a single channel, set the output_ids to zero since this input was
        # broadcasted to match the channel dimension of the other inputs
        sampling_ctx.channel_index.zero_()

    # Index the channel_index to get the correct samples for each scope
    samples = samples.gather(dim=-1, index=sampling_ctx.channel_index[instance_mask].unsqueeze(-1)).squeeze(
        -1
    )

    # Ensure, that no data is overwritten
    if data[samples_mask].isfinite().any():
        raise RuntimeError("Data already contains values at the specified mask. This should not happen.")

    # Update data inplace
    samples_mask_subset = samples_mask[instance_mask][:, module.scope.query]
    data[samples_mask] = samples[samples_mask_subset].to(data.dtype)

    return data


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CondLeafModule,
    marg_rvs: list[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Optional[LeafModule]:
    """Structural marginalization for ``NormallLayer`` objects in the ``torch`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains none of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Unaltered leaf layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    *params, = layer.retrieve_params(torch.tensor([]), dispatch_ctx)
    layer.set_params(*params)

    # Marginalized scope
    scope_marg = Scope([q for q in layer.scope.query if q not in marg_rvs])
    # Get indices of marginalized random variables in the original scope
    idxs_marg = [i for i, q in enumerate(layer.scope.query) if q in scope_marg.query]

    if len(scope_marg.query) == 0:
        return None

    # Construct new layer with marginalized scope and params
    marg_params_dict = layer.distribution.marginalized_params(idxs_marg)

    # Make sure to detach the parameters first
    marg_params_dict = {k: v.detach() for k, v in marg_params_dict.items()}

    cond_f = lambda data: marg_params_dict

    # Construct new object of the same class as the layer
    return layer.__class__(
        scope=scope_marg,
        cond_f=cond_f
    )
