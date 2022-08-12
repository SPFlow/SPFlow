"""
Created on August 10, 2022

@authors: Philipp Deibert
"""
import numpy as np
from typing import Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.layers.layer import SPNSumLayer, SPNProductLayer, SPNPartitionLayer


@dispatch(memoize=True)
def log_likelihood(sum_layer: SPNSumLayer, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_layer.children], axis=1)

    # set placeholder values
    sum_layer.set_placeholders("log_likelihood", child_lls, dispatch_ctx, overwrite=False)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return np.concatenate([log_likelihood(node, data, dispatch_ctx=dispatch_ctx) for node in sum_layer.nodes], axis=1)


@dispatch(memoize=True)
def log_likelihood(product_layer: SPNProductLayer, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in product_layer.children], axis=1)

    # set placeholder values
    product_layer.set_placeholders("log_likelihood", child_lls, dispatch_ctx, overwrite=False)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return np.concatenate([log_likelihood(node, data, dispatch_ctx=dispatch_ctx) for node in product_layer.nodes], axis=1)


@dispatch(memoize=True)
def log_likelihood(partition_layer: SPNPartitionLayer, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in partition_layer.children], axis=1)

    # set placeholder values
    partition_layer.set_placeholders("log_likelihood", child_lls, dispatch_ctx, overwrite=False)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return np.concatenate([log_likelihood(node, data, dispatch_ctx=dispatch_ctx) for node in partition_layer.nodes], axis=1)