"""
Created on November 26, 2021

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from typing import Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.nodes.leaves.parametric import TorchGaussian


@dispatch(TorchGaussian, torch.Tensor, cache=dict)
@memoize(TorchGaussian)
def log_likelihood(leaf: TorchGaussian, data: torch.Tensor, cache: Dict = {}):
    return leaf(data)