"""Contains inference methods for ``Module`` objects for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.module import Module


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Module,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Raises ``NotImplementedError`` for modules in the ``torch`` backend that have not dispatched a log-likelihood inference routine.

    Args:
        sum_node:
            Sum node to perform inference for.
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
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    raise NotImplementedError(
        f"'log_likelihood' is not defined for modules of type {type(module)}. Check if dispatched functions are correctly declared or imported."
    )


@dispatch  # type: ignore
def likelihood(
    module: Module,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Computes likelihoods for modules in the ``torch`` backend given input data.

    Likelihoods are per default computed from the infered log-likelihoods of a module.

    Args:
        modules:
            Module to perform inference for.
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
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return torch.exp(log_likelihood(module, data, check_support=check_support, dispatch_ctx=dispatch_ctx))
