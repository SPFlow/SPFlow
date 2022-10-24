"""
Created on October 24, 2021

@authors: Philipp Deibert
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.cond_node import SPNCondSumNode


@dispatch(memoize=True)
def log_likelihood(node: SPNCondSumNode, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    inputs = torch.hstack([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in node.children()])

    # retrieve value for 'weights'
    weights = node.retrieve_params(data, dispatch_ctx)

    # weight inputs in log-space
    weighted_inputs = inputs + weights.log()

    return torch.logsumexp(weighted_inputs, dim=-1, keepdims=True)