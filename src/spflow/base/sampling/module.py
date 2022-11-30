"""Contains sampling methods for ``Module`` for SPFlow in the ``base`` backend.
"""
from functools import reduce
from typing import Optional

import numpy as np

from spflow.base.structure.module import Module
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)


@dispatch  # type: ignore
def sample(
    module: Module,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Samples from modules in the ``base`` backend without any evidence.

    Samples a single instance from the module.

    Args:
        module:
            Module to sample from.
        data:
            Two-dimensional NumPy array containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional NumPy array containing the sampled values.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return sample(
        module,
        1,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )


@dispatch  # type: ignore
def sample(
    module: Module,
    n: int,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> np.ndarray:
    r"""Samples specified numbers of instances from modules in the ``base`` backend without any evidence.

    Samples a specified number of instance from the module by creating an empty two-dimensional NumPy array (i.e., filled with NaN values) of appropriate size and filling it.

    Args:
        module:
            Module to sample from.
        data:
            Two-dimensional NumPy array containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional NumPy array containing the sampled values.
        Each row corresponds to a sample.
    """
    combined_module_scope = reduce(
        lambda s1, s2: s1.join(s2), module.scopes_out
    )

    data = np.full((n, max(combined_module_scope.query) + 1), np.nan)

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    return sample(
        module,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )
