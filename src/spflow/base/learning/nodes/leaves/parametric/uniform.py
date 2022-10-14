"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.uniform import Uniform


# TODO: MLE dispatch context?


@dispatch(memoize=True)
def maximum_likelihood_estimation(leaf: Uniform, data: np.ndarray, weights: Optional[np.ndarray]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    if np.any(~leaf.check_support(data[:, leaf.scope.query])):
        raise ValueError("Encountered values outside of the support for 'Uniform'.")

    # do nothing since there are no learnable parameters
    pass