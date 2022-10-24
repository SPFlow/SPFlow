"""
Created on October 22, 2022

@authors: Philipp Deibert
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.layers.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussianLayer


@dispatch(memoize=True)
def log_likelihood(layer: CondMultivariateGaussianLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve values for 'mean','cov'
    mean_values, cov_values = layer.retrieve_params(data, dispatch_ctx)

    for node, mean, cov in zip(layer.nodes, mean_values, cov_values):
        dispatch_ctx.update_args(node, {'mean': mean, 'cov': cov})

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return torch.concat([log_likelihood(node, data, dispatch_ctx=dispatch_ctx) for node in layer.nodes], dim=1)
