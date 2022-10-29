"""
Created on October 24, 2022

@authors: Philipp Deibert
"""
from typing import Optional
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.layers.layer import (
    SPNSumLayer,
    SPNProductLayer,
    SPNPartitionLayer,
    SPNHadamardLayer,
)


@dispatch(memoize=True)  # type: ignore
def em(
    layer: SPNSumLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SPNSumLayer`` in the ``torch`` backend.

    Args:
        layer:
            Layer to perform EM step for.
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
                for child in layer.children()
            ]
        )

        # TODO: output shape ?
        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = layer.weights.data * (
            dispatch_ctx.cache["log_likelihood"][layer].grad
            * torch.exp(child_lls)
            / torch.exp(dispatch_ctx.cache["log_likelihood"][layer])
        ).sum(dim=0)

        # ----- maximization step -----
        layer.weights = expectations / expectations.sum(dim=0)

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients

    # recursively call EM on children
    for child in layer.children():
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def em(
    layer: SPNProductLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SPNProductLayer`` in the ``torch`` backend.

    Args:
        layer:
            Layer to perform EM step for.
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
    for child in layer.children():
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def em(
    layer: SPNPartitionLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SPNPartitionLayer`` in the ``torch`` backend.

    Args:
        layer:
            Layer to perform EM step for.
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
    for child in layer.children():
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def em(
    layer: SPNHadamardLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SPNHadamardLayer`` in the ``torch`` backend.

    Args:
        layer:
            Layer to perform EM step for.
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
    for child in layer.children():
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)
