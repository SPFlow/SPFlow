"""
Created on October 22, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.learning.nodes.leaves.parametric.multivariate_gaussian import maximum_likelihood_estimation
from spflow.torch.structure.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer


@dispatch(memoize=True)
def maximum_likelihood_estimation(layer: MultivariateGaussianLayer, data: torch.Tensor, weights: Optional[torch.Tensor]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None, dispatch_ctx: Optional[DispatchContext]=None) -> None:
    """TODO."""

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    if weights is None:
        weights = torch.ones((data.shape[0], layer.n_out))

    if (weights.ndim == 1 and weights.shape[0] != data.shape[0]) or \
       (weights.ndim == 2 and (weights.shape[0] != data.shape[0] or weights.shape[1] != layer.n_out)) or \
       (weights.ndim not in [1, 2]):
            raise ValueError("Number of specified weights for maximum-likelihood estimation does not match number of data points.")

    if weights.ndim == 1:
        # broadcast weights
        weights = torch.unsqueeze(weights, 1).repeat(layer.n_out, 1)

    for node, node_weights in zip(layer.nodes, weights.T):
        maximum_likelihood_estimation(node, data, node_weights, bias_correction=bias_correction, nan_strategy=nan_strategy, dispatch_ctx=dispatch_ctx)



@dispatch(memoize=True)
def em(layer: MultivariateGaussianLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # call EM on internal nodes
    for node in layer.nodes:
        em(node, data, dispatch_ctx=dispatch_ctx)