from spflow.modules.leaf import Normal, Bernoulli
from tests.fixtures import auto_set_test_seed, auto_set_test_device, auto_set_test_device
import unittest
from spflow.modules.rat import RatSPN
from itertools import product

from spflow import InvalidParameterCombinationError, ScopeError
from spflow.meta import Scope
import pytest
from spflow.meta import SamplingContext
from spflow.meta.dispatch import init_default_sampling_context, init_default_dispatch_context
from spflow import log_likelihood, sample, marginalize, sample_with_evidence
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import Sum, ElementwiseProduct
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_leaf, make_data
import torch
from spflow.modules import leaf

depth = [1, 3]
n_region_nodes = [1, 5]
num_leaves = [1, 6]
num_repetitions = [1, 7]
n_root_nodes = [1, 4]
outer_product = [True, False]
split_halves = [True, False]
leaf_cls_values = [
    # leaf.Bernoulli,
    # leaf.Binomial,
    # leaf.Categorical,
    # leaf.Exponential,
    # leaf.Gamma,
    # leaf.Geometric,
    # leaf.Hypergeometric,
    # leaf.LogNormal,
    # leaf.NegativeBinomial,
    leaf.Normal,
    # leaf.Poisson,
    # leaf.Uniform,
]
params = list(
    product(
        leaf_cls_values,
        depth,
        n_region_nodes,
        num_leaves,
        num_repetitions,
        n_root_nodes,
        outer_product,
        split_halves,
    )
)


def make_rat_spn(
    leaf_cls,
    depth,
    n_region_nodes,
    num_leaves,
    num_repetitions,
    n_root_nodes,
    num_features,
    outer_product,
    split_halves,
):
    depth = depth
    n_region_nodes = n_region_nodes
    num_leaves = num_leaves
    num_repetitions = num_repetitions
    n_root_nodes = n_root_nodes
    num_features = num_features

    leaf_layer = make_leaf(
        cls=leaf_cls, out_channels=num_leaves, out_features=num_features, num_repetitions=num_repetitions
    )

    model = RatSPN(
        leaf_modules=[leaf_layer],
        n_root_nodes=n_root_nodes,
        n_region_nodes=n_region_nodes,
        num_repetitions=num_repetitions,
        depth=depth,
        outer_product=outer_product,
        split_halves=split_halves,
    )
    return model


@pytest.mark.parametrize(
    "leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_halves ", params
)
def test_log_likelihood(leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_halves):
    num_features = 64
    module = make_rat_spn(
        leaf_cls=leaf_cls,
        depth=d,
        n_region_nodes=region_nodes,
        num_leaves=leaves,
        num_repetitions=num_reps,
        n_root_nodes=root_nodes,
        num_features=num_features,
        outer_product=outer_product,
        split_halves=split_halves,
    )
    assert len(module.scope) == num_features
    data = make_data(cls=leaf_cls, out_features=num_features, n_samples=10)
    ctx = init_default_dispatch_context()
    # data = data.unsqueeze(1).repeat(1,3,1)
    lls = log_likelihood(module, data, dispatch_ctx=ctx)

    assert lls.shape == (data.shape[0], module.out_features, module.out_channels)


@pytest.mark.parametrize(
    "leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_halves ", params
)
def test_sample(leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_halves):
    n_samples = 100
    num_features = 64
    module = make_rat_spn(
        leaf_cls=leaf_cls,
        depth=d,
        n_region_nodes=region_nodes,
        num_leaves=leaves,
        num_repetitions=num_reps,
        n_root_nodes=root_nodes,
        num_features=num_features,
        outer_product=outer_product,
        split_halves=split_halves,
    )
    for i in range(module.out_channels):
        data = torch.full((n_samples, num_features), torch.nan)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
        mask = torch.full((n_samples, module.out_features), True)
        repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )
        samples = sample(module, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "region_nodes, leaves, num_reps, root_nodes, outer_product, split_halves ",
    list(product(n_region_nodes, num_leaves, num_repetitions, n_root_nodes, outer_product, split_halves)),
)
def test_multidistribution_input(region_nodes, leaves, num_reps, root_nodes, outer_product, split_halves):
    out_features_1 = 8
    out_features_2 = 10
    depth = 2

    scope_1 = Scope(list(range(0, out_features_1)))
    scope_2 = Scope(list(range(out_features_1, out_features_1 + out_features_2)))

    cls_1 = Normal
    cls_2 = Bernoulli

    module_1 = make_leaf(cls=cls_1, out_channels=leaves, scope=scope_1, num_repetitions=num_reps)
    data_1 = make_data(cls=cls_1, out_features=out_features_1, n_samples=5)

    module_2 = make_leaf(cls=cls_2, out_channels=leaves, scope=scope_2, num_repetitions=num_reps)
    data_2 = make_data(cls=cls_2, out_features=out_features_2, n_samples=5)

    data = torch.cat((data_1, data_2), dim=1)

    model = RatSPN(
        leaf_modules=[module_1, module_2],
        n_root_nodes=root_nodes,
        n_region_nodes=region_nodes,
        num_repetitions=num_reps,
        depth=depth,
        outer_product=outer_product,
        split_halves=split_halves,
    )

    lls = log_likelihood(model, data)

    assert lls.shape == (data.shape[0], model.out_features, model.out_channels)

    repetition_idx = torch.zeros((1,), dtype=torch.long)
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, num_samples=1)
    sampling_ctx.repetition_idx = repetition_idx
    samples = sample(model, sampling_ctx=sampling_ctx)

    assert samples.shape == (1, out_features_1 + out_features_2)
