# -*- coding: utf-8 -*-
"""Contains sampling methods for ``CondBinomialLayer`` leaves for SPFlow in the ``base`` backend.
"""
import numpy as np
from typing import Optional
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.contexts.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.base.structure.layers.leaves.parametric.cond_binomial import (
    CondBinomialLayer,
)
from spflow.base.sampling.module import sample


@dispatch  # type: ignore
def sample(
    layer: CondBinomialLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> np.ndarray:
    r"""Samples from ``CondBinomialLayer`` leaves in the ``base`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Samples missing values proportionally to its probability mass function (PMF).

    Args:
        layer:
            Leaf layer to sample from.
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
        Two-dimensional NumPy array containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.

    Raises:
        ValueError: Sampling from invalid number of outputs.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # make sure no over-lapping scopes are being sampled
    layer_scopes = layer.scopes_out

    # retrieve value for 'p'
    p_values = layer.retrieve_params(data, dispatch_ctx)

    for node, p in zip(layer.nodes, p_values):
        dispatch_ctx.update_args(node, {"p": p})

    # sample
    for output_ids in np.unique(sampling_ctx.output_ids, axis=0):
        if len(output_ids) == 0:
            output_ids = list(range(layer.n_out))

        if not Scope.all_pairwise_disjoint(
            [layer_scopes[id] for id in output_ids]
        ):
            raise ValueError(
                "Sampling from non-pairwise-disjoint scopes for instances is not allowed."
            )

    # all product nodes are over (all) children
    for node_id, instances in sampling_ctx.group_output_ids(layer.n_out):
        sample(
            layer.nodes[node_id],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(instances, [[] for _ in instances]),
        )

    return data
