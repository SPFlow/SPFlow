"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(leaf: Hypergeometric, data: np.ndarray, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    if np.any(~leaf.check_support(data[:, leaf.scope.query])):
        raise ValueError("Encountered values outside of the support for 'Hypergeometric'.")

    # do nothing since there are no learnable parameters
    pass