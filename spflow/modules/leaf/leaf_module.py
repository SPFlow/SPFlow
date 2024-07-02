from abc import ABC
from typing import Optional, Union
from collections.abc import Callable

import torch
from torch import Tensor

from spflow.distributions.distribution import Distribution
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.modules.module import Module
from spflow.utils.leaf import apply_nan_strategy


class LeafModule(Module, ABC):
    def __init__(self, scope: Union[Scope, list[int]], out_channels: int = None):
        r"""Initializes ``Normal`` leaf node.

        Args:
            scope: Scope object or list of ints specifying the scope of the distribution.
            out_channels: Number of output channels.
        """
        super().__init__()

        # Convert list to Scope object
        if isinstance(scope, list):
            scope = Scope(scope)

        self.scope = scope.copy()
        self._out_channels = out_channels

    @property
    def distribution(self) -> Distribution:
        return self._distribution

    @distribution.setter
    def distribution(self, distribution: Distribution):
        self._distribution = distribution

    @property
    def out_features(self) -> int:
        return len(self.scope.query)

    @property
    def out_channels(self) -> int:
        return self.distribution.out_channels


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
    leaf: LeafModule,
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
    leaf: LeafModule,
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

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    # apply NaN strategy
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # Forward to the actual distribution
    leaf.distribution.maximum_likelihood_estimation(scope_data, weights, bias_correction)


@dispatch  # type: ignore
def sample(
    leaf: LeafModule,
    data: Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from the leaf nodes in the ``torch`` backend given potential evidence.

    Samples missing values proportionally to its probability distribution function (PDF).

    Args:
        leaf:
            Leaf node to sample from.
        data:
            Two-dimensional PyTorch tensor containing potential evidence.
            Each row corresponds to a sample.
        is_mpe:
            Boolean value indicating whether or not to perform maximum a posteriori estimation (MPE).
            Defaults to False.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
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
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    inverse_scope_query = list(filter(lambda x: x not in leaf.scope.query, range(data.shape[1])))
    # marg_ids = torch.isnan(data[:, leaf.scope.query])
    marg_ids = torch.isnan(data)
    marg_ids[:, inverse_scope_query] = False

    instance_ids_mask = torch.zeros(data.shape[0], 1, device=leaf.device, dtype=torch.bool)
    instance_ids_mask[sampling_ctx.instance_ids] = True

    sampling_mask = marg_ids & instance_ids_mask
    n_samples = torch.sum(sampling_mask.sum(1) > 0)  # count number of rows which have at least one true value
    if is_mpe:
        samples = leaf.distribution.mode()

        # Add batch dimension
        samples = samples.unsqueeze(0).repeat(n_samples, *([1] * (samples.dim())))
    else:
        samples = leaf.distribution.sample(n_samples=n_samples)

    # Use output_ids from sampling context to index into the correct outputs for each scope
    # I.e.: For each sample and for each
    assert samples.shape[0] == sampling_ctx.output_ids.shape[0]
    assert samples.shape[0] == data.shape[0]

    if leaf.out_channels > 1:
        # Index the output_ids to get the correct samples for each scope
        # Output_ids should usually be defined by some module that is the parent of this layer
        # assert samples.shape[1] == sampling_ctx.output_ids.shape[1]#assert samples.shape[-1] == sampling_ctx.output_ids.shape[1]
        samples = samples.gather(dim=-1, index=sampling_ctx.output_ids.unsqueeze(-1)).squeeze(-1)

    samples = samples.view(samples.shape[0], samples.shape[1])

    # Set data at correct scope
    sampling_mask_at_scope = sampling_mask[:, leaf.scope.query]
    data[marg_ids] = samples[sampling_mask_at_scope].type(data.dtype)

    return data


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: LeafModule,
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

    # Construct new object of the same class as the layer
    return layer.__class__(
        scope=scope_marg,
        **marg_params_dict,
    )
