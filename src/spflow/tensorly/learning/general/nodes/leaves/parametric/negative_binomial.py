"""Contains learning methods for ``NegativeBinomial`` nodes for SPFlow in the ``base`` backend.
"""
from typing import Callable, Optional, Union

import tensorly as tl
from ......utils.helper_functions import tl_isnan, tl_isclose

from spflow.tensorly.structure.general.nodes.leaves.parametric.negative_binomial import (
    NegativeBinomial,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: NegativeBinomial,
    data: tl.tensor,
    weights: Optional[tl.tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``NegativeBinomial`` node parameters in the ``base`` backend.

    Estimates the success probability :math:`p` of a Negative Binomial distribution from data, as follows:

    .. math::

        p^{\*}=\frac{n\sum_{i=1}^N w_i}{\sum_{i=1}^{N}w_i(x_i+n)}

    where
        - :math:`n` is the number of successes
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

    The number of successes is fixed and will not be estimated.
    Weights are normalized to sum up to :math:`N`.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one-dimensional NumPy array containing non-negative weights for all data samples.
            Must match number of samples in ``data``.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Has no effect for ``NegativeBinomial`` nodes.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another NumPy array of same size.
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

    if weights is None:
        weights = tl.ones(data.shape[0])

    if tl.ndim(weights) != 1 or tl.shape(weights)[0] != tl.shape(data)[0]:
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )

    # reshape weights
    weights = tl.reshape(weights,(-1, 1))

    if check_support:
        if tl.any(~leaf.check_support(scope_data, is_scope_data=True)):
            raise ValueError("Encountered values outside of the support for 'NegativeBinomial'.")

    # NaN entries (no information)
    nan_mask = tl_isnan(scope_data)

    if tl.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")

    if nan_strategy is None and tl.any(nan_mask):
        raise ValueError(
            "Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended."
        )

    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            # simply ignore missing data
            scope_data = scope_data[~nan_mask.squeeze(1)]
            weights = weights[~nan_mask.squeeze(1)]
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'NegativeBinomial'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
        # TODO: how to handle weights?
    elif nan_strategy is not None:
        raise ValueError(
            f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}."
        )

    # normalize weights to sum to n_samples
    weights /= tl.sum(weights)  / tl.shape(scope_data)[0]

    # total (weighted) number of successes
    n_success = tl.sum(weights)  * leaf.n

    # total number of trials
    n_total = tl.sum(weights * (scope_data + leaf.n))

    # estimate (weighted) success probability
    p_est = n_success / n_total

    # edge case: if prob. 1 (or 0), set to smaller (or larger) value
    if tl_isclose(p_est, 0.0):
        p_est = 1e-8
    elif tl_isclose(p_est, 1):
        p_est = 1 - 1e-8

    # set parameters of leaf node
    leaf.set_params(n=leaf.n, p=p_est)
