"""Contains sampling methods for ``CondNegativeBinomial`` nodes for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.cond_negative_binomial import (
    CondNegativeBinomial,
)


@dispatch  # type: ignore
def sample(
    leaf: CondNegativeBinomial,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> torch.Tensor:
    r"""Samples from ``CondNegativeBinomial`` nodes in the ``torch`` backend given potential evidence.

    Samples missing values proportionally to its probability mass function (PMF).

    Args:
        leaf:
            Leaf node to sample from.
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
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    # retrieve value for 'p'
    p = leaf.retrieve_params(data, dispatch_ctx)

    marg_ids = (torch.isnan(data[:, leaf.scope.query]) == len(leaf.scope.query)).squeeze(1)

    instance_ids_mask = torch.zeros(data.shape[0], device=leaf.device)
    instance_ids_mask[sampling_ctx.instance_ids] = 1

    sampling_ids = marg_ids.to(leaf.device) & instance_ids_mask.bool().to(leaf.device)

    data[sampling_ids, leaf.scope.query] = leaf.dist(p=p).sample((sampling_ids.sum(),)).squeeze(-1).to(leaf.device)

    return data
