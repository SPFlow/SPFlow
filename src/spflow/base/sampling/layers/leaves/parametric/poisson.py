"""
Created on August 14, 2022

@authors: Philipp Deibert
"""
import numpy as np
from typing import Optional
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.base.structure.layers.leaves.parametric.poisson import PoissonLayer
from spflow.base.sampling.module import sample


@dispatch
def sample(layer: PoissonLayer, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> np.ndarray:
    """TODO"""
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # make sure no over-lapping scopes are being sampled
    layer_scopes = layer.scopes_out

    for output_ids in np.unique(sampling_ctx.output_ids, axis=0):
        if len(output_ids) == 0:
            output_ids = list(range(layer.n_out))

        if not Scope.all_pairwise_disjoint([layer_scopes[id] for id in output_ids]):
            raise ValueError("Sampling from non-pairwise-disjoint scopes for instances is not allowed.")

    # all product nodes are over (all) children
    for node_id, instances in sampling_ctx.group_output_ids(layer.n_out):
        sample(layer.nodes[node_id], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(instances, [[] for _ in instances]))

    return data