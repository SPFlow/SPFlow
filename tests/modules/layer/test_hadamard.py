import unittest
from typing import Union
from spflow.modules.layer.leaf.normal import Normal as NormalLayer
from spflow.modules.node.leaf.normal import Normal as NormalNode
from spflow.meta.dispatch import SamplingContext
from spflow.meta.dispatch import init_default_sampling_context
from spflow.meta.data import Scope
from spflow import log_likelihood, sample, marginalize
from tests.modules.node.leaf.utils import make_normal_leaf, make_normal_data
from spflow.learn import expectation_maximization
from spflow.modules.layer import HadamardLayer
import pytest
import torch

# Constants
TOTAL_NODES = 2
NUM_LEAVES = 2
NUM_SCOPES = 3
TOTAL_SCOPES = 5

def make_hadamard(inputs=None):
    if inputs is None:
        inputs = make_normal_leaf("layer", num_scopes=NUM_SCOPES, num_leaves=NUM_LEAVES)
    return HadamardLayer(inputs=[inputs])

def test_log_likelihood():
    hadamard_layer = make_hadamard()
    data = make_normal_data(dim=TOTAL_SCOPES)
    lls = log_likelihood(hadamard_layer, data)
    assert lls.shape == (data.shape[0], hadamard_layer.event_shape[-1])

def test_log_likelihood_result():
    mean = torch.tensor([[0.0, 1.0],[-1.0, 2.0], [-2.0, 3.0]])
    std = torch.tensor([[1.0, 1.0],[1.0, 1.0], [1.0, 1.0]])
    leaf_layer = make_normal_leaf("layer", num_scopes=NUM_SCOPES, num_leaves=NUM_LEAVES, mean=mean, std=std)
    hadamard_layer = make_hadamard(inputs=leaf_layer)
    data = make_normal_data(num_samples=1,dim=TOTAL_SCOPES)
    lls = log_likelihood(hadamard_layer, data)
    leaf_layer_ll = log_likelihood(leaf_layer, data)[0]
    s1 = leaf_layer_ll[0,0] + leaf_layer_ll[1,0] + leaf_layer_ll[2,0]
    s2 = leaf_layer_ll[0,0] + leaf_layer_ll[1,0] + leaf_layer_ll[2,1]
    s3 = leaf_layer_ll[0,0] + leaf_layer_ll[1,1] + leaf_layer_ll[2,0]
    s4 = leaf_layer_ll[0,0] + leaf_layer_ll[1,1] + leaf_layer_ll[2,1]
    s5 = leaf_layer_ll[0,1] + leaf_layer_ll[1,0] + leaf_layer_ll[2,0]
    s6 = leaf_layer_ll[0,1] + leaf_layer_ll[1,0] + leaf_layer_ll[2,1]
    s7 = leaf_layer_ll[0,1] + leaf_layer_ll[1,1] + leaf_layer_ll[2,0]
    s8 = leaf_layer_ll[0,1] + leaf_layer_ll[1,1] + leaf_layer_ll[2,1]
    expected_lls = torch.tensor([s1,s2, s3, s4, s5, s6, s7, s8]).reshape(1,8)
    assert torch.allclose(lls, expected_lls, atol=1e-5)

def test_sample(n_samples=100):
    hadamard_layer = make_hadamard()
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)
    for i in range(hadamard_layer.event_shape[-1]):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        samples = sample(hadamard_layer, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:,hadamard_layer.scope.query]
        assert torch.isfinite(samples_query).all()


def test_expectation_maximization():
    hadamard_layer = make_hadamard()
    data = make_normal_data(dim=TOTAL_SCOPES)
    expectation_maximization(hadamard_layer, data, max_steps=10)

"""
def test_constructor():
    with pytest.raises(ValueError):
        # Product nodes must have disjunct input scopes
        make_hadamard(inputs=[[make_normal_leaf(scope=0), make_normal_leaf(scope=0)]])
        # input has to be a list of lists
        make_hadamard(inputs=[make_normal_leaf(scope=0), make_normal_leaf(scope=1)])
"""

@pytest.mark.parametrize("prune", [True, False])
def test_marginalize(prune):

    # default scope: [1, 2, 3]
    hadamard_layer = make_hadamard()

    # Marginalize first scope
    marginalized_hadamard_layer = marginalize(hadamard_layer, [1], prune=prune)

    # Scope query should not contain [1]
    assert len(set(marginalized_hadamard_layer.scope.query).intersection([1])) == 0

    marginalized_hadamard_layer2 = marginalize(marginalized_hadamard_layer, [2], prune=prune)

    if prune:
        # If pruning, NormalLayer should be returned
        assert isinstance(marginalized_hadamard_layer2, NormalLayer)
    else:
        # Else HadamardLayer should be returned
        assert isinstance(marginalized_hadamard_layer2, HadamardLayer)

    # Scope query should not contain [1, 2]
    assert len(set(marginalized_hadamard_layer2.scope.query).intersection([1,2])) == 0

    marginalized_hadamard_layer3 = marginalize(marginalized_hadamard_layer2, [3], prune=prune)

    # Scope query should contain  None
    if NUM_SCOPES == 3:
        assert marginalized_hadamard_layer3 is None
    else:
        assert len(set(marginalized_hadamard_layer3.scope.query).intersection([1, 2, 3])) == 0

if __name__ == "__main__":
    unittest.main()
