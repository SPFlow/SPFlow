"""
Created on September 23, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.learning.nodes.leaves.parametric.exponential import maximum_likelihood_estimation
from spflow.base.structure.layers.leaves.parametric.exponential import ExponentialLayer


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(leaf: ExponentialLayer, data: np.ndarray, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""
    for node in leaf.nodes:
        maximum_likelihood_estimation(node, data, bias_correction=bias_correction, nan_strategy=nan_strategy)