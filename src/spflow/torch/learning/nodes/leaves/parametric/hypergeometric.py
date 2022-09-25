"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(leaf: Hypergeometric, data: torch.Tensor, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    if torch.any(~leaf.check_support(data[:, leaf.scope.query])):
        raise ValueError("Encountered values outside of the support for 'Hypergeometric'.")

    # do nothing since there are no learnable parameters
    pass