# -*- coding: utf-8 -*-
"""Contains sampling methods for modules for SPFlow in the ``base`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.base.structure.module import Module, NestedModule

import numpy as np
from typing import Optional
from functools import reduce


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
    sampling_ctx: Optional[DispatchContext] = None,
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


@dispatch  # type: ignore
def sample(
    placeholder: NestedModule.Placeholder,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> np.ndarray:
    r"""Samples from a placeholder modules in the ``base`` with potential evidence.

    Samples from the actual inputs represented by the placeholder module.

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
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # dictionary to hold the
    sampling_ids_per_child = [([], []) for _ in placeholder.host.children]

    for instance_id, output_ids in zip(
        sampling_ctx.instance_ids, sampling_ctx.output_ids
    ):
        # convert ids to actual child and output ids of host module
        child_ids_actual, output_ids_actual = placeholder.input_to_output_ids(
            output_ids
        )

        for child_id in np.unique(child_ids_actual):
            sampling_ids_per_child[child_id][0].append(instance_id)
            sampling_ids_per_child[child_id][1].append(
                np.array(output_ids_actual)[
                    child_ids_actual == child_id
                ].tolist()
            )

    # sample from children
    for child_id, (instance_ids, output_ids) in enumerate(
        sampling_ids_per_child
    ):
        if len(instance_ids) == 0:
            continue
        sample(
            placeholder.host.children[child_id],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(instance_ids, output_ids),
        )

    return data
