# -*- coding: utf-8 -*-
"""Contains inference methods for SPN-like nodes for SPFlow in the ``base`` backend.
"""
import numpy as np
from scipy.special import logsumexp  # type: ignore
from typing import Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode


@dispatch(memoize=True)  # type: ignore
def log_likelihood(sum_node: SPNSumNode, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """Computes log-likelihoods for SPN-like sum nodes in the ``base`` backend given input data.

    Log-likelihood for sum node is the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_node:
            Sum node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_node.children], axis=1)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return logsumexp(child_lls, b=sum_node.weights, axis=1, keepdims=True)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(product_node: SPNProductNode, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """Computes log-likelihoods for SPN-like product nodes in the ``base`` backend given input data.

    Log-likelihood for product node is the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        product_node:
            Product node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in product_node.children], axis=1)

    # multiply child log-likelihoods together (sum in log-space)
    return child_lls.sum(axis=1, keepdims=True)