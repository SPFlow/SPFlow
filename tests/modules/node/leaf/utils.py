from spflow import log_likelihood, sample
import torch
from spflow.modules.leaf_module import LeafModule


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
