"""Contains learning methods for ``CategoricalLayer`` leaves for SPFlow in the ``torch`` backend.
"""
from typing import Callable, Optional, Union

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.layers.leaves.parametric.categorical import (
    CategoricalLayer,
)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    layer: CategoricalLayer,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``CategoricalLayer`` leaves' parameters in the ``torch`` backend.

    Estimates the success probabilities :math:`p` of each Categorical distribution from data, as follows:

    .. math::

        p_k^{\*}=\frac{\sum_{i=1}^{N}w_iI_k(x_i)}{\sum_{i=1}^N w_i}


    where
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set
        - :math:`I_k(x)` is an indicator function of k, returning 1 if x belongs to category k, otherwise 0

    Weights are normalized to sum up to :math:`N` per row.

    Args:
        layer:
            Layer to estimate parameters of.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one- or two-dimensional PyTorch tensor containing non-negative weights for all data samples and nodes.
            Must match number of samples in ``data``.
            If a one-dimensional PyTorch tensor is given, the weights are broadcast to all nodes.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Not relevant for ``CategoricalLayer`` leaves.
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
    scope_data = torch.hstack([data[:, scope.query] for scope in layer.scopes_out])

    if weights is None:
        weights = torch.ones(data.shape[0], layer.n_out)

    if (
        (weights.ndim == 1 and weights.shape[0] != data.shape[0])
        or (weights.ndim == 2 and (weights.shape[0] != data.shape[0] or weights.shape[1] != layer.n_out))
        or (weights.ndim not in [1, 2])
    ):
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )

    if weights.ndim == 1:
        # broadcast weights
        weights = weights.repeat(layer.n_out, 1).T

    if check_support:
        if torch.any(~layer.check_support(scope_data, is_scope_data=True)):
            raise ValueError("Encountered values outside of the support for 'BernoulliLayer'.")

    # NaN entries (no information)
    nan_mask = torch.isnan(scope_data)

    # check if any columns (i.e., data for a output scope) contain only NaN values
    if torch.any(nan_mask.sum(dim=0) == scope_data.shape[0]):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data for a specified scope.")

    if nan_strategy is None and torch.any(nan_mask):
        raise ValueError(
            "Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended."
        )

    if nan_strategy is None:
        # exlude NaNs
        weights = weights * ~nan_mask
    elif (isinstance(nan_strategy, str) and nan_strategy == "ignore"):
        scope_data = torch.nan_to_num(scope_data, nan=0.0)
    elif isinstance(nan_strategy, Callable):
        # transform data according to provided callback
        scope_data = nan_strategy(scope_data)
    else:
        raise ValueError(f"Unknown strategy {nan_strategy} of type {type(nan_strategy)} for handling missing values for 'CategoricalLayer'.")
    

    # TODO: weights slow down MLE for categoricals, so implement weight-less MLE using torch.unique(<data>, return_counts=True)
    

    # normalize weights to sum to n_samples
    weights /= weights.sum(dim=0) / scope_data.shape[0]

    # total (weighted) number of instances
    n_total = weights.sum(dim=0)

    # count (weighted) number of total successes
    n_success = (weights * scope_data).sum(dim=0)

    # estimate (weighted) success probability
    p_est = n_success / n_total

    p_est = []
    for column in range(scope_data.shape[1]):
        p_k_est = []
        for cat in range(len(torch.unique(data))):
            cat_indices = scope_data[:, column] == cat
            cat_data = cat_indices.float()
            #cat_weights = weights[cat_indices] # this leads to array of different shape and cant be used in that way here
            cat_est = torch.sum(weights[:, column] * cat_data)
            cat_est /= n_total[column]
            p_k_est.append(cat_est)
        p_est.append(p_k_est)

    # set parameters of leaf node
    layer.set_params(p=p_est)


@dispatch(memoize=True)  # type: ignore
def em(
    layer: CategoricalLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``CategoricalLayer`` in the ``torch`` backend.

    Args:
        layer:
            Leaf layer to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = dispatch_ctx.cache["log_likelihood"][layer].grad
        # normalize expectations for better numerical stability
        expectations /= expectations.sum(dim=0)

        # ----- maximization step -----

        # update parameters through maximum weighted likelihood estimation
        maximum_likelihood_estimation(
            layer,
            data,
            weights=expectations,
            bias_correction=False,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )

    # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients
