"""
Created on November 26, 2021

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from typing import Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.nodes import TorchProductNode, TorchSumNode


@dispatch(TorchProductNode, torch.Tensor, cache=dict)
@memoize(TorchProductNode)
def log_likelihood(node: TorchProductNode, data: torch.Tensor, cache: Dict = {}) -> torch.Tensor:
    inputs = torch.hstack([log_likelihood(child, data, cache=cache) for child in node.children()])  # type: ignore
    return node.forward(inputs)


@dispatch(TorchSumNode, torch.Tensor, cache=dict)
@memoize(TorchSumNode)
def log_likelihood(node: TorchSumNode, data: torch.Tensor, cache: Dict = {}) -> torch.Tensor:
    inputs = torch.hstack([log_likelihood(child, data, cache=cache) for child in node.children()])  # type: ignore
    return node.forward(inputs)
