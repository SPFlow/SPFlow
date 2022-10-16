"""
Created on October 13, 2022

@authors: Philipp Deibert
"""
from typing import Optional
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.node import SPNSumNode, SPNProductNode


@dispatch(memoize=True)
def em(node: SPNSumNode, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----
        child_lls = torch.hstack([dispatch_ctx.cache['log_likelihood'][child] for child in node.children()])

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = node.weights.data * (dispatch_ctx.cache['log_likelihood'][node].grad * torch.exp(child_lls) / torch.exp(dispatch_ctx.cache['log_likelihood'][node])).sum(dim=0)

        # ----- maximization step -----
        node.weights = expectations/expectations.sum()

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients

    # recursively call EM on children
    for child in node.children():
        em(child, data, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)
def em(node: SPNProductNode, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # recursively call EM on children
    for child in node.children():
        em(child, data, dispatch_ctx=dispatch_ctx)