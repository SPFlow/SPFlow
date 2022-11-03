# -*- coding: utf-8 -*-
"""Contains inference methods for SPN-like conditional nodes for SPFlow in the ``torch`` backend.
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.spn.nodes.cond_sum_node import SPNCondSumNode


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    node: SPNCondSumNode,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Computes log-likelihoods for conditional SPN-like sum node given input data in the ``torch`` backend.

    Log-likelihood for sum node is the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_node:
            Sum node to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    inputs = torch.hstack(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in node.children()
        ]
    )

    # retrieve value for 'weights'
    weights = node.retrieve_params(data, dispatch_ctx)

    # weight inputs in log-space
    weighted_inputs = inputs + weights.log()

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return torch.logsumexp(weighted_inputs, dim=-1, keepdims=True)
