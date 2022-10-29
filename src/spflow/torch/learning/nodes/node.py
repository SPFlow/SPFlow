"""
Created on October 13, 2022

@authors: Philipp Deibert
"""
from typing import Optional
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.nodes.node import SPNSumNode, SPNProductNode


@dispatch(memoize=True)  # type: ignore
def em(
    node: SPNSumNode,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SPNSumNode`` in the ``torch`` backend.

    Args:
        node:
            Node to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----
        child_lls = torch.hstack(
            [
                dispatch_ctx.cache["log_likelihood"][child]
                for child in node.children()
            ]
        )

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = node.weights.data * (
            dispatch_ctx.cache["log_likelihood"][node].grad
            * torch.exp(child_lls)
            / torch.exp(dispatch_ctx.cache["log_likelihood"][node])
        ).sum(dim=0)

        # ----- maximization step -----
        node.weights = expectations / expectations.sum()

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients

    # recursively call EM on children
    for child in node.children():
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def em(
    node: SPNProductNode,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SPNProductNode`` in the ``torch`` backend.

    Args:
        node:
            Node to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # recursively call EM on children
    for child in node.children():
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)
