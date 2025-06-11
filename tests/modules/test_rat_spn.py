from spflow.modules.leaf import Normal
from tests.fixtures import auto_set_test_seed, auto_set_test_device, auto_set_test_device
import unittest
from spflow.modules.rat.rat_spn import RatSPN
from itertools import product

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.data import Scope
import pytest
from spflow.meta.dispatch import init_default_sampling_context, init_default_dispatch_context, SamplingContext
from spflow import log_likelihood, sample, marginalize, sample_with_evidence
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import Sum, ElementwiseProduct
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_leaf
import torch

depth = [2,5]
n_region_nodes = [1,5]#10#10
num_leaves = [1,6]#10
num_repetitions = [1,7]#2#5
n_root_nodes = [1,4]#10
params = list(product(depth, n_region_nodes, num_leaves, num_repetitions, n_root_nodes))




def make_rat_spn(depth, n_region_nodes, num_leaves, num_repetitions, n_root_nodes, num_features):
    depth = depth
    n_region_nodes = n_region_nodes
    num_leaves = num_leaves
    num_repetitions = num_repetitions
    n_root_nodes = n_root_nodes
    num_features = num_features

    random_variables = list(range(num_features))
    scope = Scope(random_variables)

    normal_layer = Normal(scope=scope, out_channels=num_leaves, num_repetitions=num_repetitions)

    model = RatSPN(
        leaf_modules=[normal_layer],
        n_root_nodes=n_root_nodes,
        n_region_nodes=n_region_nodes,  # 30
        num_repetitions=num_repetitions,  # 1,
        depth=depth,
        outer_product=True,
        split_halves=False,
    )
    return model

@pytest.mark.parametrize("d, region_nodes, leaves, num_reps, root_nodes ", params)
def test_log_likelihood(d, region_nodes, leaves, num_reps, root_nodes):
    num_features = 64
    module = make_rat_spn(
        depth=d,
        n_region_nodes=region_nodes,
        num_leaves=leaves,
        num_repetitions=num_reps,
        n_root_nodes=root_nodes,
        num_features=num_features,

    )
    data = make_normal_data(out_features=num_features)
    ctx = init_default_dispatch_context()
    lls = log_likelihood(module, data, dispatch_ctx=ctx)

    assert lls.shape == (data.shape[0], module.out_features, module.out_channels)

@pytest.mark.parametrize("d, region_nodes, leaves, num_reps, root_nodes ", params)
def test_sample(d, region_nodes, leaves, num_reps, root_nodes):
    n_samples = 100
    num_features = 64
    module = make_rat_spn(
        depth=d,
        n_region_nodes=region_nodes,
        num_leaves=leaves,
        num_repetitions=num_reps,
        n_root_nodes=root_nodes,
        num_features=num_features,

    )
    for i in range(module.out_channels):
        data = torch.full((n_samples, num_features), torch.nan)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
        mask = torch.full((n_samples, module.out_features), True)
        repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
        samples = sample(module, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()
