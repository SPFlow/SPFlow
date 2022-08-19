"""
Created on August 05, 2022

@authors: Philipp Deibert
"""
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.leaves.parametric.binomial import Binomial

from typing import Optional
import numpy as np


@dispatch(memoize=True)
def log_likelihood(node: Binomial, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope.query]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data).sum(axis=-1).astype(bool)

    # create masked based on distribution's support
    valid_ids = node.check_support(data[~marg_ids]).squeeze(1)

    # TODO: suppress checks
    if not all(valid_ids):
        raise ValueError(
            f"Encountered data instances that are not in the support of the Binomial distribution."
        )

    # compute probabilities for all non-marginalized instances
    probs[~marg_ids] = node.dist.logpmf(k=data[~marg_ids])

    return probs