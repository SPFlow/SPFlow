"""
Created on September 23, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.base.learning.nodes.leaves.parametric.negative_binomial import maximum_likelihood_estimation
from spflow.base.structure.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer


@dispatch(memoize=True)
def maximum_likelihood_estimation(layer: NegativeBinomialLayer, data: np.ndarray, weights: Optional[np.ndarray]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None, dispatch_ctx: Optional[DispatchContext]=None) -> None:
    """TODO."""

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    if weights is None:
        weights = np.ones((data.shape[0], layer.n_out))

    if (weights.ndim == 1 and weights.shape[0] != data.shape[0]) or \
       (weights.ndim == 2 and (weights.shape[0] != data.shape[0] or weights.shape[1] != layer.n_out)) or \
       (weights.ndim not in [1, 2]):
            raise ValueError("Number of specified weights for maximum-likelihood estimation does not match number of data points.")

    if weights.ndim == 1:
        # broadcast weights
        weights = np.expand_dims(weights, 1).repeat(layer.n_out, 1)

    for node, node_weights in zip(layer.nodes, weights.T):
        maximum_likelihood_estimation(node, data, node_weights, bias_correction=bias_correction, nan_strategy=nan_strategy, dispatch_ctx=dispatch_ctx)