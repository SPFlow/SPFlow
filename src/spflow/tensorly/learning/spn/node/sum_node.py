"""Contains learning methods for SPN-like sum nodes for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.tensorly.structure.spn.node.sum_node import SumNode


@dispatch(memoize=True)  # type: ignore
def em(
    node: SumNode,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SumNode`` in the ``torch`` backend.

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
        child_lls = torch.hstack([dispatch_ctx.cache["log_likelihood"][child] for child in node.children])

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
    for child in node.children:
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)
