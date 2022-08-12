"""
Created on August 08, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.sampling.module import sample

import numpy as np
from typing import Optional


@dispatch
def sample(node: SPNSumNode, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> np.ndarray:

    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # compute log likelihoods of data instances (TODO: only compute for relevant instances? might clash with cashed values or cashing in general)
    child_lls = np.concatenate(
        [log_likelihood(child, data, dispatch_ctx=dispatch_ctx) for child in node.children], axis=1
    )

    # take child likelihoods into account when sampling
    sampling_weights = node.weights + child_lls[sampling_ctx.instance_ids]

    # sample branch for each instance id
    # this solution is based on a trick described here: https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035#34190035    
    cum_sampling_weights = sampling_weights.cumsum(axis=1)
    random_choices = np.random.rand(sampling_weights.shape[0], 1)
    branches = (cum_sampling_weights < random_choices).sum(axis=1)

    # group sampled branches
    for branch in np.unique(branches):
        # group instances by sampled branch
        branch_instance_ids = np.array(sampling_ctx.instance_ids)[branches == branch].tolist()

        # get corresponding child and output id for sampled branch
        child_ids, output_ids = node.input_to_output_ids([branch])

        # sample from child module
        sample(
            node.children[child_ids[0]], data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(branch_instance_ids, [[output_ids[0]] for _ in range(len(branch_instance_ids))])
        )

    return data


@dispatch
def sample(node: SPNProductNode, data: np.ndarray, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> np.ndarray:

    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # sample from all child outputs
    for child in node.children:
        data = sample(child, data, dispatch_ctx=dispatch_ctx, sampling_ctx=SamplingContext(sampling_ctx.instance_ids, [list(range(child.n_out)) for _ in sampling_ctx.instance_ids]))

    return data