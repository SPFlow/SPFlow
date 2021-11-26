import torch
from multipledispatch import dispatch  # type: ignore
from typing import Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.rat import (
    TorchRatSpn,
    _TorchPartitionLayer,
    _TorchRegionLayer,
    _TorchLeafLayer,
)


@dispatch(_TorchPartitionLayer, torch.Tensor, cache=dict)
@memoize(_TorchPartitionLayer)
def log_likelihood(layer: _TorchPartitionLayer, data: torch.Tensor, cache: Dict = {}):
    inputs = [log_likelihood(child, data, cache=cache) for child in layer.children()]
    return layer(inputs)


@dispatch(_TorchRegionLayer, torch.Tensor, cache=dict)
@memoize(_TorchRegionLayer)
def log_likelihood(layer: _TorchRegionLayer, data: torch.Tensor, cache: Dict = {}):
    inputs = [log_likelihood(child, data, cache=cache) for child in layer.children()]
    return layer(inputs)


@dispatch(_TorchLeafLayer, torch.Tensor, cache=dict)
@memoize(_TorchLeafLayer)
def log_likelihood(layer: _TorchLeafLayer, data: torch.Tensor, cache: Dict = {}):
    inputs = [log_likelihood(child, data, cache=cache) for child in layer.children()]
    return layer(inputs)


@dispatch(TorchRatSpn, torch.Tensor, cache=dict)
@memoize(TorchRatSpn)
def log_likelihood(rat_spn: TorchRatSpn, data: torch.Tensor, cache: Dict = {}):
    input = log_likelihood(rat_spn.root_region, data, cache=cache)
    return rat_spn(input)
