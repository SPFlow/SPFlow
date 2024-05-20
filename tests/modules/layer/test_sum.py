import unittest

import pytest
from spflow.meta.dispatch import init_default_sampling_context
from spflow.meta.data import Scope
from spflow.modules.layer.leaf.normal import Normal as NormalLayer
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from spflow.modules.layer import SumLayer
from tests.modules.node.leaf.utils import make_normal_leaf, make_normal_data
import torch

# Constants
TOTAL_NODES = 2
NUM_LEAVES = 2
NUM_SCOPES = 4
TOTAL_SCOPES = 5


def make_sum(num_nodes=3, weights=None, inputs=None):
    if inputs is None:
        inputs = make_normal_leaf("layer", num_scopes=NUM_SCOPES, num_leaves=NUM_LEAVES)
    return SumLayer(n_nodes=num_nodes, inputs=inputs, weights=weights)


def test_log_likelihood():
    num_nodes = 3
    sum_layer = make_sum(num_nodes=num_nodes)
    data = make_normal_data(dim=TOTAL_SCOPES)
    lls = log_likelihood(sum_layer, data)
    assert lls.shape == (data.shape[0], NUM_SCOPES, num_nodes)


def test_log_likelihood_result():
    num_nodes = 3
    # if NUM_SCOPES != 3 mean and std have to be adjusted
    NUM_SCOPES = 3

    mean = torch.tensor([[0.0, 1.0], [-1.0, 2.0], [-2.0, 3.0]])
    std = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    leaf_layer = make_normal_leaf("layer", num_scopes=NUM_SCOPES, num_leaves=NUM_LEAVES, mean=mean, std=std)
    weights = torch.ones((num_nodes, NUM_SCOPES, NUM_LEAVES))
    sum_layer = make_sum(weights= weights, inputs=leaf_layer)
    data = make_normal_data(num_samples=1,dim=TOTAL_SCOPES)
    lls = log_likelihood(sum_layer, data)
    leaf_layer_ll = log_likelihood(leaf_layer, data)
    """
    leaf_layer_ll = [[[-0.9620, -1.7554], [-1.6137, -9.6501], [-4.2174, -3.8752]]]
    """
    weighted_leaf_layer_ll = leaf_layer_ll + torch.log(torch.tensor(0.5))
    expected_lls = torch.logsumexp(weighted_leaf_layer_ll, dim=-1).T.unsqueeze(0)
    assert torch.allclose(lls, expected_lls, atol=1e-5)


def test_sample(n_samples=100):
    num_nodes = 3
    sum_layer = make_sum(num_nodes=num_nodes)
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)
    for i in range(num_nodes):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        samples = sample(sum_layer, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:,sum_layer.scope.query]
        assert torch.isfinite(samples_query).all()


def test_expectation_maximization():
    sum_layer = make_sum()
    data = make_normal_data(dim=TOTAL_SCOPES)
    expectation_maximization(sum_layer, data, max_steps=10)


def test_weights():
    num_nodes = 3
    weights = torch.ones((num_nodes, NUM_SCOPES, NUM_LEAVES))
    sum_layer = make_sum(num_nodes=num_nodes, weights=weights)
    assert torch.allclose(sum_layer.weights.sum(dim=-1), torch.tensor(1.0))
    assert torch.allclose(sum_layer.log_weights, sum_layer.weights.log())

"""
def test_constructor():
    # Check invalid parameters
    with pytest.raises(ValueError):
        sum_layer = make_sum(weights=[-0.5, 0.5, 1.0])

    with pytest.raises(ValueError):
        sum_layer = make_sum(weights=[-0.5, 1.5])

    with pytest.raises(ValueError):
        sum_layer = make_sum(weights=[0.5, 0.2])

    with pytest.raises(ValueError):
        sum_layer = make_sum(weights=[0.2, 0.5])

    with pytest.raises(ValueError):
        sum_layer = make_sum(inputs=[make_normal_leaf(scope=0), make_normal_leaf(scope=1)])
"""

@pytest.mark.parametrize("prune", [True, False])
def test_marginalize(prune):

    # default scope: [1, 2, 3]
    sum_layer = make_sum()

    # Marginalize first scope
    marginalized_sum_layer = marginalize(sum_layer, [1], prune=prune)

    # Scope query should not contain [1]
    assert len(set(marginalized_sum_layer.scope.query).intersection([1])) == 0

    marginalized_sum_layer2 = marginalize(marginalized_sum_layer, [2], prune=prune)

    # Scope query should not contain [1, 2]
    assert len(set(marginalized_sum_layer2.scope.query).intersection([1, 2])) == 0

    # Scope query should contain  None
    marginalized_sum_layer3 = marginalize(marginalized_sum_layer2, [3], prune=prune)

    if NUM_SCOPES == 3:
        assert marginalized_sum_layer3 is None
    else:
        assert len(set(marginalized_sum_layer3.scope.query).intersection([1, 2, 3])) == 0

if __name__ == "__main__":
    unittest.main()
