"""Contains learning methods for ``Exponential`` nodes for SPFlow in the ``base`` backend.
"""
from typing import Callable, Optional, Union

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_isclose, T

from spflow.tensorly.structure.general.nodes.leaves.parametric.exponential import (
    Exponential,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Exponential,
    data: T,
    weights: Optional[T] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Exponential`` node parameters in the ``base`` backend.

    Estimates the rate parameter :math:`l` of an Exponential distribution from data, as follows:

    .. math::

        l^{\*}=\frac{\sum_{i=1}^N w_i}{\sum_{i=1}^N w_ix_i}

    or

    .. math::

        l^{\*}=\frac{(\sum_{i=1}^N w_i)-1}{\sum_{i=1}^N w_ix_i}

    if bias correction is used, where
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

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
        weights = tl.ones(tl.shape(data)[0])

    if tl.ndim(weights) != 1 or tl.shape(weights)[0] != tl.shape(data)[0]:
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )

    # reshape weights
    weights = tl.reshape(weights,(-1, 1))

    if check_support:
        if tl.any(~leaf.check_support(scope_data, is_scope_data=True)):
            raise ValueError("Encountered values outside of the support for 'Exponential'.")

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
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'Exponential'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
        # TODO: how to handle weights?
    elif nan_strategy is not None:
        raise ValueError(
            f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}."
        )

    # normalize weights to sum to n_samples
    weights /= tl.sum(weights) / tl.shape(scope_data)[0]

    # total number of instances
    n_total = tl.sum(weights)

    if bias_correction:
        n_total -= 1

    # cummulative evidence
    cum_rate = tl.sum(weights * scope_data)

    # estimate rate parameter
    l_est = n_total / cum_rate

    # edge case: if rate 0, set to larger value (should not happen, but just in case)
    if tl_isclose(l_est, 0):
        l_est = 1e-8

    # set parameters of leaf node
    leaf.set_params(l=l_est)
