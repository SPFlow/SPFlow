"""
Created on September 25, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.layers.leaves.parametric.hypergeometric import HypergeometricLayer


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(layer: HypergeometricLayer, data: torch.Tensor, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = torch.hstack([data[:, scope.query] for scope in layer.scopes_out])

    if torch.any(~layer.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'HypergeometricLayer'.")

    # do nothing since there are no learnable parameters
    pass