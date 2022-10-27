# -*- coding: utf-8 -*-
"""Contains inference methods for SPN-like conditional layers for SPFlow in the ``torch`` backend.
"""
import torch
import numpy as np
from typing import Optional
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.layers.cond_layer import SPNCondSumLayer


@dispatch(memoize=True)  # type: ignore
def log_likelihood(sum_layer: SPNCondSumLayer, data: torch.Tensor, check_support: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """Computes log-likelihoods for conditional SPN-like sum layers given input data in the ``torch`` backend.

    Log-likelihoods for sum nodes are the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_layer:
            Sum layer to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve value for 'weights'
    weights = sum_layer.retrieve_params(data, dispatch_ctx)

    # compute child log-likelihoods
    child_lls = torch.concat([log_likelihood(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx) for child in sum_layer.children()], dim=1)

    weighted_lls = child_lls.unsqueeze(1) + weights.log()

    return torch.logsumexp(weighted_lls, dim=-1)