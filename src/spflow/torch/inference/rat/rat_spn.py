"""
Created on November 26, 2021

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from typing import Optional, Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.rat import (
    TorchRatSpn,
    _TorchPartitionLayer,
    _TorchRegionLayer,
    _TorchLeafLayer,
)


@dispatch(_TorchPartitionLayer, torch.Tensor, cache=dict)
@memoize(_TorchPartitionLayer)
def log_likelihood(layer: _TorchPartitionLayer, data: torch.Tensor, cache: Optional[Dict] = None):

    if(cache is None):
        cache = {}

    inputs = [log_likelihood(child, data, cache=cache) for child in layer.children()]

    batch_size = inputs[0].shape[0]

    out_batch = []
    input_batch = [[inp[i] for inp in inputs] for i in range(batch_size)]

    # multiply cartesian products (sum in log space) for each entry in batch
    for inputs_batch in input_batch:
        out_batch.append(torch.cartesian_prod(*inputs_batch).sum(dim=1))

    out = torch.vstack(out_batch)  # type: ignore

    return out


@dispatch(_TorchRegionLayer, torch.Tensor, cache=dict)
@memoize(_TorchRegionLayer)
def log_likelihood(layer: _TorchRegionLayer, data: torch.Tensor, cache: Optional[Dict] = None):

    if(cache is None):
        cache = {}

    inputs = torch.hstack([log_likelihood(child, data, cache=cache) for child in layer.children()])  # type: ignore

    # broadcast inputs per output node and weight them in log-space
    weighted_inputs = inputs.unsqueeze(1) + layer.weights.log()  # type: ignore

    return torch.logsumexp(weighted_inputs, dim=-1)


@dispatch(_TorchLeafLayer, torch.Tensor, cache=dict)
@memoize(_TorchLeafLayer)
def log_likelihood(layer: _TorchLeafLayer, data: torch.Tensor, cache: Optional[Dict] = None):

    if(cache is None):
        cache = {}

    inputs = [log_likelihood(child, data, cache=cache) for child in layer.children()]

    return torch.hstack(inputs)  # type: ignore


@dispatch(TorchRatSpn, torch.Tensor, cache=dict)
@memoize(TorchRatSpn)
def log_likelihood(rat_spn: TorchRatSpn, data: torch.Tensor, cache: Optional[Dict] = None):

    if(cache is None):
        cache = {}

    inputs = log_likelihood(rat_spn.root_region, data, cache=cache)

    # broadcast inputs per output node and weight them
    weighted_inputs = inputs.unsqueeze(1) + rat_spn.root_node_weights.log()  # type: ignore

    return torch.logsumexp(weighted_inputs, dim=-1)
