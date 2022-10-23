"""
Created on October 22, 2022

@authors: Philipp Deibert
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer
from spflow.torch.inference.nodes.leaves.parametric.multivariate_gaussian import log_likelihood


@dispatch(memoize=True)
def log_likelihood(layer: MultivariateGaussianLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return torch.concat([log_likelihood(node, data, dispatch_ctx=dispatch_ctx) for node in layer.nodes], dim=1)