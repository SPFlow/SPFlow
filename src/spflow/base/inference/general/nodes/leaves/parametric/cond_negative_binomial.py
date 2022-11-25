"""Contains inference methods for ``CondNegativeBinomial`` nodes for SPFlow in the ``base`` backend.
"""
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.general.nodes.leaves.parametric.cond_negative_binomial import (
    CondNegativeBinomial,
)

from typing import Optional
import numpy as np


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    node: CondNegativeBinomial,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``CondNegativeBinomial`` node given input data in the ``base`` backend.

    Log-likelihood for ``CondNegativeBinomial`` is given by the logarithm of its probability mass function (PMF):

    .. math::

        \log(\text{PMF}(k)) = \log(\binom{k+n-1}{n-1}p^n(1-p)^k)

    where
        - :math:`k` is the number of failures
        - :math:`n` is the maximum number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the distribution.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve value for 'p'
    p = node.retrieve_params(data, dispatch_ctx)

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope.query]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data).sum(axis=-1).astype(bool)

    if check_support:
        # create masked based on distribution's support
        valid_ids = node.check_support(
            data[~marg_ids], is_scope_data=True
        ).squeeze(1)

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the NegativeBinomial distribution."
            )

    # compute probabilities for all non-marginalized instances
    probs[~marg_ids] = node.dist(p=p).logpmf(k=data[~marg_ids])

    return probs
