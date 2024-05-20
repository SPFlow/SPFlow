from spflow import log_likelihood, sample
from typing import Union
import torch
from spflow.meta.data import Scope
from spflow.modules.leaf_module import LeafModule
from spflow.modules.layer.leaf.normal import Normal as NormalLayer
from spflow.modules.node.leaf.normal import Normal as NormalNode
from spflow.modules.module import Module


def evaluate_log_likelihood(node: LeafModule, data: torch.Tensor):
    lls = log_likelihood(node, data, check_support=True)
    assert lls.shape == (data.shape[0], len(node.scope.query), node.event_shape[1])
    assert torch.isfinite(lls).all()

def evaluate_samples(node: LeafModule, data: torch.Tensor, is_mpe: bool, sampling_ctx):
    samples = sample(node, data, is_mpe=is_mpe, check_support=True, sampling_ctx=sampling_ctx)
    assert samples.shape == data.shape
    s_query = samples[:, node.scope.query]
    assert s_query.shape == (data.shape[0], len(node.scope.query))
    assert torch.isfinite(s_query).all()

def make_normal_leaf(module_type: str, num_scopes=None, num_leaves=None, mean=None, std=None) -> Union[NormalNode, NormalLayer]:
    """
    Create a Normal leaf node.

    Args:
        module_type: Type of the module.
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
    """
    if module_type == "node":
        mean = mean if mean is not None else torch.randn(1)
        std = std if std is not None else torch.rand(1) + 1e-8
        scope = Scope([1])
        return NormalNode(scope=scope, mean=mean, std=std)
    elif module_type == "layer":
        assert (num_scopes is not None) and (num_leaves is not None)
        mean = mean if mean is not None else torch.randn(num_scopes, num_leaves)
        std = std if std is not None else torch.rand(num_scopes, num_leaves) + 1e-8
        scope = Scope(list(range(1, num_scopes + 1)))
        return NormalLayer(scope=scope, mean=mean, std=std)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")

def make_normal_data(mean=0.0, std=1.0, num_samples=10, dim=2):
    torch.manual_seed(0)
    return torch.randn(num_samples, dim) * std + mean
