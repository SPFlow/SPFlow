# -*- coding: utf-8 -*-
"""Contains inference methods for ``NestedModule`` objects for SPFlow in the ``torch`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.torch.structure.nested_module import NestedModule

from typing import Optional
import torch


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    nesting_module: NestedModule.Placeholder,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Raises ``LookupError`` for placeholder-modules in the ``torch`` backend.

    The log-likelihoods for placeholder-modules should be set in the dispatch context cache by the host module.
    This method is only called if the cache is not filled properly, due to memoization.

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
    raise LookupError(
        "Log-likelihood values for 'NestedModule.Placeholder' must not have been found in dispatch cache. Check if these are correctly set by the host module."
    )
