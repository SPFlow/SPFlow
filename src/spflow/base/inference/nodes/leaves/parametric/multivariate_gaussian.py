"""
Created on August 05, 2022

@authors: Philipp Deibert
"""
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian

from typing import Optional
import numpy as np
from scipy.stats import multivariate_normal


@dispatch(memoize=True)
def log_likelihood(node: MultivariateGaussian, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope.query]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data)

    # number of marginalized random variables per instance
    n_marg = marg_ids.sum(axis=-1)

    # in case of partially marginalized instances
    if any((n_marg > 0) & (n_marg < len(node.scope.query))):
        raise ValueError(f"Partial marginalization not yet supported for MultivariateGaussian.")

    # create masked based on distribution's support
    valid_ids = node.check_support(data[~n_marg.astype(bool)])

    # TODO: suppress checks
    if not valid_ids.all():
        raise ValueError(
            f"Encountered data instances that are not in the support of the MultivariateGaussian distribution."
        )

    # compute probabilities for all non-marginalized instances
    probs[~n_marg.astype(bool), 0] = multivariate_normal.logpdf(x=data[~n_marg.astype(bool)], mean=node.mean, cov=node.cov)

    return probs