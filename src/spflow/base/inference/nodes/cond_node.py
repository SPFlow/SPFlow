# -*- coding: utf-8 -*-
"""Contains inference methods for SPN-like conditional nodes for SPFlow in the ``base`` backend.
"""
import numpy as np
from scipy.special import logsumexp  # type: ignore
from typing import List, Type, Dict, Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.cond_node import SPNCondSumNode


@dispatch(memoize=True)  # type: ignore
def log_likelihood(sum_node: SPNCondSumNode, data: np.ndarray, check_support: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """Computes log-likelihoods for conditional SPN-like sum node given input data in the ``base`` backend.

    Log-likelihood for sum node is the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_node:
            Sum node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve value for 'weights'
    weights = sum_node.retrieve_params(data, dispatch_ctx)

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx) for child in sum_node.children], axis=1)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return logsumexp(child_lls, b=weights, axis=1, keepdims=True)