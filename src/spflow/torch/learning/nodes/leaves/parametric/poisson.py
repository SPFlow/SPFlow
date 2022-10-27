# -*- coding: utf-8 -*-
"""Contains learning methods for ``Poisson`` nodes for SPFlow in the ``torch`` backend.
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.leaves.parametric.poisson import Poisson


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(leaf: Poisson, data: torch.Tensor, weights: Optional[torch.Tensor]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None, dispatch_ctx: Optional[DispatchContext]=None) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Poisson`` node parameters in the ``torch`` backend.

    Estimates the rate parameter :math:`l` of a Poisson distribution from data, as follows:

    .. math::

        l^{\*}=\frac{1}{\sum_{i=1}^N w_i}\sum_{i=1}^N w_ix_i

    where
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

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
            Has no effect for ``Poisson`` nodes.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another PyTorch tensor of same size.
            Defaults to None.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        ValueError: Invalid arguments.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if weights is None:
        weights = torch.ones(data.shape[0])

    if weights.ndim != 1 or weights.shape[0] != data.shape[0]:
        raise ValueError("Number of specified weights for maximum-likelihood estimation does not match number of data points.")

    # reshape weights
    weights = weights.reshape(-1, 1)

    if torch.any(~leaf.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'Poisson'.")

    # NaN entries (no information)
    nan_mask = torch.isnan(scope_data)

    if torch.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")

    if nan_strategy is None and torch.any(nan_mask):
        raise ValueError("Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended.")
    
    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            # simply ignore missing data
            scope_data = scope_data[~nan_mask]
            weights = weights[~nan_mask.squeeze(1)]
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'Poisson'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
        # TODO: how to handle missing data?
    elif nan_strategy is not None:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    # normalize weights to sum to n_samples
    weights /= weights.sum() / data.shape[0]

    # total number of instances
    n_total = weights.sum(dtype=torch.get_default_dtype())

    # estimate rate parameter from data
    l_est = (weights * scope_data.type(torch.get_default_dtype())).sum() / n_total

    # edge case: if rate 0, set to larger value (should not happen, but just in case)
    if torch.isclose(l_est, torch.tensor(0.0)):
        l_est = 1e-8

    # set parameters of leaf node
    leaf.set_params(l=l_est)


@dispatch(memoize=True)  # type: ignore
def em(leaf: Poisson, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:
    """Performs a single expectation maximizaton (EM) step for ``Poisson`` in the ``torch`` backend.

    Args:
        leaf:
            Leaf node to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = dispatch_ctx.cache['log_likelihood'][leaf].grad
        # normalize expectations for better numerical stability
        expectations /= expectations.sum()

        # ----- maximization step -----

        # update parameters through maximum weighted likelihood estimation
        maximum_likelihood_estimation(leaf, data, weights=expectations.squeeze(1), bias_correction=False, dispatch_ctx=dispatch_ctx)

    # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients