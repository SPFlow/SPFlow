"""
Created on August 17, 2022

@authors: Philipp Deibert
"""
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.torch.structure.layers.leaves.parametric.hypergeometric import HypergeometricLayer
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample

import torch
import numpy as np
from typing import Optional
import itertools


@dispatch
def sample(layer: HypergeometricLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    unique_output_signatures = set(frozenset(l) for l in sampling_ctx.output_ids)

    # make sure that no scopes are overlapping
    # TODO: suppress check
    for output_ids in unique_output_signatures:
        if len(output_ids) == 0:
            output_ids = list(range(layer.n_out))

        if not Scope.all_pairwise_disjoint([layer.scopes_out[id] for id in output_ids]):
            raise ValueError("Sampling from output with non-pair-wise disjoint scopes is not permitted for 'HypergeometricLayer'.")

    # group sampling instances by node
    for node_id, instances in sampling_ctx.group_output_ids(layer.n_out):

        node_scope = layer.scopes_out[node_id]

        # TODO: what to do in case of instance ids that are already specified (i.e. not nan)?
        marg_ids = (torch.isnan(data[:, node_scope.query]) == len(node_scope.query)).squeeze(1)

        instance_ids_mask = torch.zeros(data.shape[0])
        instance_ids_mask[torch.tensor(instances)] = 1

        sampling_mask = marg_ids & instance_ids_mask.bool().to(layer.N.device)
        sampling_ids = torch.where(sampling_mask)[0]

        # TODO: may be inefficient
        # create random permutations of N elements
        rand_perm = torch.argsort(torch.rand(sampling_ids.shape[0], layer.N[node_id]), dim=1)
    
        # assuming that first M indices are the M objects of interest, count how many of these indices were "drawn" in the first n draws (with replacement since all indices are unique per row)
        data[torch.meshgrid(sampling_ids, torch.tensor(node_scope.query), indexing='ij')] = (rand_perm[:, :layer.n[node_id]] < layer.M[node_id]).sum(dim=1).type(torch.get_default_dtype()).unsqueeze(1)

    return data