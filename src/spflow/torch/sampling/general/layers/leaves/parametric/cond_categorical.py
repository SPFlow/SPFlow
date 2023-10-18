"""Contains sampling methods for ``CondCategoricalLayer`` leaves for SPFlow in the ``torch`` backend.
"""
import itertools
from typing import Optional

import numpy as np
import torch

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample
from spflow.torch.structure.general.layers.leaves.parametric.cond_categorical import (
    CondCategoricalLayer,
)


@dispatch  # type: ignore
def sample(
    layer: CondCategoricalLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> torch.Tensor:
    r"""Samples from ``CondCategoricalLayer`` leaves in the ``torch`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Samples missing values proportionally to its probability mass function (PMF).

    Args:
        layer:
            Leaf layer to sample from.
        data:
            Two-dimensional PyTorch tensor containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional PyTorch tensor containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.

    Raises:
        ValueError: Sampling from invalid number of outputs.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    unique_output_signatures = {frozenset(l) for l in sampling_ctx.output_ids}

    # retrieve value for 'p'
    k, p = layer.retrieve_params(data, dispatch_ctx)

    # make sure that no scopes are overlapping
    # : suppress check
    for output_ids in unique_output_signatures:
        if len(output_ids) == 0:
            output_ids = list(range(layer.n_out))

        if not Scope.all_pairwise_disjoint([layer.scopes_out[id] for id in output_ids]):
            raise ValueError(
                "Sampling from output with non-pair-wise disjoint scopes is not permitted for 'CategoricalLayer'."
            )

    # group sampling instances by node
    for node_id, instances in sampling_ctx.group_output_ids(layer.n_out):

        node_scope = layer.scopes_out[node_id]

        marg_ids = (torch.isnan(data[:, node_scope.query]) == len(node_scope.query)).squeeze(1)

        instance_ids_mask = torch.zeros(data.shape[0])
        instance_ids_mask[torch.tensor(instances)] = 1

        sampling_mask = marg_ids & instance_ids_mask.bool().to(p.device)
        sampling_ids = torch.where(sampling_mask)[0]

        data[torch.meshgrid(sampling_ids, torch.tensor(node_scope.query), indexing="ij")] = (
            layer.dist(p=p, node_ids=[node_id]).sample((sampling_mask.sum(),)).type(torch.float).to(p.device)
        )

    return data
