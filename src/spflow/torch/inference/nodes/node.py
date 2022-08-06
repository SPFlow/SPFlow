"""
Created on November 26, 2021

@authors: Philipp Deibert
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import SPNProductNode, SPNSumNode


@dispatch(memoize=True)
def log_likelihood(node: SPNProductNode, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    inputs = torch.hstack([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in node.children()])

    # return product (sum in log space)
    return torch.sum(inputs, dim=-1, keepdims=True)


@dispatch(memoize=True)
def log_likelihood(node: SPNSumNode, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    inputs = torch.hstack([log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in node.children()])

    # weight inputs in log-space
    weighted_inputs = inputs + node.weights.log()

    return torch.logsumexp(weighted_inputs, dim=-1, keepdims=True)