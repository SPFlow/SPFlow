"""Contains inference methods for ``Categorical`` nodes for SPFlow in the ``base`` backend.
"""
from typing import Optional

import numpy as np

from spflow.base.structure.general.nodes.leaves.parametric.categorical import Categorical
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    node: Categorical,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``Categorical`` node in the ``base`` backend given input data.

    Log-likelihood for ``Categorical`` is given by the logarithm of its probability mass function (PMF):

    .. math::

        \log(\text{PMF}(k))=\log(p_k)

    where
        - :math:`k` is an integer associated with a category
        - :math:`p_k` is the success probability of the associated category k in :math:`[0,1]`

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
    
    Raises:
        ValueError: Data outside of support.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope.query]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data).sum(axis=-1).astype(bool)

    if check_support:
        # create masked based on distribution's support
        valid_ids = node.check_support(data[~marg_ids], is_scope_data=True).squeeze(1)

        if not all(valid_ids):
            raise ValueError(f"Encountered data instances that are not in the support of the Categorical distribution.")

    # create one-hot-encoding to compute probs in scipy's multinomial
    data_copy = data[~marg_ids].copy()
    ohe_data = []
    for row in data_copy:
        for k in row:
            ohe_element = np.zeros((node.k, 1))
            ohe_element[k] = 1
            ohe_data.append(ohe_element.flatten())
    ohe_data = np.array(ohe_data)

    # compute probabilities for all non-marginalized instances, skip if data is empty to prevent scipy errors
    if ohe_data.ndim and ohe_data.size:
        probs[~marg_ids] = node.dist.logpmf(x=ohe_data).reshape(-1, 1)

    return probs
