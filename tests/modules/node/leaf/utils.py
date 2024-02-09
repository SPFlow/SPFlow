from spflow import log_likelihood
import torch
from spflow.modules.leaf_module import LeafModule


def evaluate_log_likelihood(node: LeafModule, data: torch.Tensor):
    lls = log_likelihood(node, data, check_support=True)
    assert lls.shape == (data.shape[0], len(node.scope.query), node.event_shape[1])
    assert torch.isfinite(lls).all()
