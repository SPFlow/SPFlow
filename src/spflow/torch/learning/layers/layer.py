"""
Created on October 24, 2022

@authors: Philipp Deibert
"""
from typing import Optional
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.layers.layer import SPNSumLayer, SPNProductLayer, SPNPartitionLayer, SPNHadamardLayer


@dispatch(memoize=True)
def em(node: SPNSumLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----
        child_lls = torch.hstack([dispatch_ctx.cache['log_likelihood'][child] for child in node.children()])

        # TODO: output shape ?
        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = node.weights.data * (dispatch_ctx.cache['log_likelihood'][node].grad * torch.exp(child_lls) / torch.exp(dispatch_ctx.cache['log_likelihood'][node])).sum(dim=0)

        # ----- maximization step -----
        node.weights = expectations/expectations.sum(dim=0)

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients

    # recursively call EM on children
    for child in node.children():
        em(child, data, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)
def em(node: SPNProductLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # recursively call EM on children
    for child in node.children():
        em(child, data, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)
def em(node: SPNPartitionLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # recursively call EM on children
    for child in node.children():
        em(child, data, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)
def em(node: SPNHadamardLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # recursively call EM on children
    for child in node.children():
        em(child, data, dispatch_ctx=dispatch_ctx)