"""
Created on August 05, 2022

@authors: Philipp Deibert

This file provides the inference methods for basic nodes.
"""
import numpy as np
from scipy.special import logsumexp  # type: ignore
from typing import List, Type, Dict, Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode


@dispatch(memoize=True)
def log_likelihood(sum_node: SPNSumNode, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_node.children], axis=1)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return logsumexp(child_lls, b=sum_node.weights, axis=1, keepdims=True)


@dispatch(memoize=True)
def log_likelihood(product_node: SPNProductNode, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in product_node.children], axis=1)

    # multiply child log-likelihoods together (sum in log-space)
    return child_lls.sum(axis=1, keepdims=True)