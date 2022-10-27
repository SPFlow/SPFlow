# -*- coding: utf-8 -*-
"""Contains inference methods for SPN-like conditional layers for SPFlow in the ``base`` backend.
"""
import numpy as np
from typing import Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.layers.cond_layer import SPNCondSumLayer
from spflow.base.inference.nodes.cond_node import log_likelihood


@dispatch(memoize=True)  # type: ignore
def log_likelihood(sum_layer: SPNCondSumLayer, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """Computes log-likelihoods for conditional SPN-like sum layers given input data in the ``base`` backend.

    Log-likelihoods for sum nodes are the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_layer:
            Sum layer to perform inference for.
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

    # retrieve value for 'weights'
    weights = sum_layer.retrieve_params(data, dispatch_ctx)

    for node, w in zip(sum_layer.nodes, weights):
        dispatch_ctx.update_args(node, {'weights': w})

    # compute child log-likelihoods
    child_lls = np.concatenate([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in sum_layer.children], axis=1)

    # set placeholder values
    sum_layer.set_placeholders("log_likelihood", child_lls, dispatch_ctx, overwrite=False)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return np.concatenate([log_likelihood(node, data, dispatch_ctx=dispatch_ctx) for node in sum_layer.nodes], axis=1)