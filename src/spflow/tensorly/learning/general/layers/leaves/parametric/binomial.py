"""Contains learning methods for ``BinomialLayer`` leaves for SPFlow in the ``base`` backend.
"""
from typing import Callable, Optional, Union

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_unsqueeze, tl_repeat, T

from spflow.tensorly.learning.general.nodes.leaves.parametric.binomial import (
    maximum_likelihood_estimation,
)
from spflow.tensorly.structure.general.layers.leaves.parametric.binomial import (
    BinomialLayer,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    layer: BinomialLayer,
    data: T,
    weights: Optional[T] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``BinomialLayer`` leaves' parameters in the ``base`` backend.

    Estimates the success probabilities :math:`p` of each Binomial distribution from data, as follows:

    .. math::

        p^{\*}=\frac{1}{n\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_ix_i

    where
        - :math:`n` is the number of i.i.d. Bernoulli trials
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

    The numbers of i.i.d. Bernoulli trials are fixed and will not be estimated.
    Weights are normalized to sum up to :math:`N` per row.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one- or two-dimensional NumPy array containing non-negative weights for all data samples and nodes.
            Must match number of samples in ``data``.
            If a one-dimensional NumPy array is given, the weights are broadcast to all nodes.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Not relevant for ``BinomialLayer`` leaves.
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

    if weights is None:
        weights = tl.ones((tl.shape(data)[0], layer.n_out))

    if (
        (tl.ndim(weights) == 1 and tl.shape(weights)[0] != tl.shape(data)[0])
        or (tl.ndim(weights) == 2 and (tl.shape(weights)[0] != tl.shape(data)[0] or tl.shape(weights)[1] != layer.n_out))
        or (tl.ndim(weights) not in [1, 2])
    ):
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )

    if tl.ndim(weights) == 1:
        # broadcast weights
        weights = tl_repeat(tl_unsqueeze(weights, 1), repeats=layer.n_out, axis=1)

    for node, node_weights in zip(layer.nodes, weights.T):
        maximum_likelihood_estimation(
            node,
            data,
            node_weights,
            bias_correction=bias_correction,
            nan_strategy=nan_strategy,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )