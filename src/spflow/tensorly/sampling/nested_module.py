"""Contains sampling methods for ``NestedModule`` for SPFlow in the ``base`` backend.
"""
from typing import Optional

import tensorly as tl
from ..utils.helper_functions import tl_unique

from spflow.tensorly.structure.nested_module import NestedModule
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
    placeholder: NestedModule.Placeholder,
    data: tl.tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> tl.tensor:
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
    sampling_ctx = init_default_sampling_context(sampling_ctx, tl.shape(data)[0])

    # dictionary to hold the
    sampling_ids_per_child = [([], []) for _ in placeholder.host.children]

    for instance_id, output_ids in zip(sampling_ctx.instance_ids, sampling_ctx.output_ids):
        # convert ids to actual child and output ids of host module
        child_ids_actual, output_ids_actual = placeholder.input_to_output_ids(output_ids)

        for child_id in tl_unique(child_ids_actual):
            sampling_ids_per_child[child_id][0].append(instance_id)
            sampling_ids_per_child[child_id][1].append(
                tl.tensor(output_ids_actual)[child_ids_actual == child_id].tolist()
            )

    # sample from children
    for child_id, (instance_ids, output_ids) in enumerate(sampling_ids_per_child):
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
