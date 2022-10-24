"""
Created on October 24, 2022

@authors: Philipp Deibert
"""
import numpy as np
from scipy.special import logsumexp  # type: ignore
from typing import List, Type, Dict, Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.cond_node import SPNCondSumNode


@dispatch(memoize=True)
def log_likelihood(sum_node: SPNCondSumNode, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve value for 'weights'
    weights = sum_node.retrieve_params(data, dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_node.children], axis=1)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return logsumexp(child_lls, b=weights, axis=1, keepdims=True)