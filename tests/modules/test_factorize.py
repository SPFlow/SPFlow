from itertools import product

import pytest
import torch

from spflow.exceptions import StructureError
from spflow.learn import expectation_maximization
from spflow.meta.data.scope import Scope
from spflow.modules.rat import Factorize
from spflow.utils.sampling_context import SamplingContext
from spflow.utils.sampling_context import init_default_sampling_context
from tests.utils.leaves import DummyLeaf, make_data, make_leaf, make_normal_data, make_normal_leaf

# Constants
in_channels_values = [1, 3]
out_features_values = [4, 8]
num_repetitions = [5]
depth_values = [1, 2]
params = list(product(in_channels_values, out_features_values, num_repetitions, depth_values))


def make_product(in_channels=None, out_features=None, inputs=None, num_repetitions=None, depth=1):
    if inputs is None:
        inputs = make_normal_leaf(
            out_features=out_features, out_channels=in_channels, num_repetitions=num_repetitions
        )
    return Factorize(inputs=[inputs], depth=depth, num_repetitions=num_repetitions)


@pytest.mark.parametrize("in_channels,out_features,num_reps,depth", params)
def test_log_likelihood(in_channels: int, out_features: int, num_reps, depth):
    factorization_layer = make_product(
        in_channels=in_channels, out_features=out_features, num_repetitions=num_reps, depth=depth
    )
    data = make_normal_data(out_features=out_features)
    lls = factorization_layer.log_likelihood(data)
    if num_reps is None:
        assert lls.shape == (
            data.shape[0],
            factorization_layer.out_features,
            factorization_layer.out_channels,
        )
    else:
        assert lls.shape == (
            data.shape[0],
            factorization_layer.out_features,
            factorization_layer.out_channels,
            num_reps,
        )


@pytest.mark.parametrize("in_channels,out_features,num_reps, depth", params)
def test_sample(in_channels: int, out_features: int, num_reps, depth):
    n_samples = 10
    factorization_layer = make_product(
        in_channels=in_channels, out_features=out_features, num_repetitions=num_reps, depth=depth
    )

    data = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(
        low=0, high=factorization_layer.out_channels, size=(n_samples, factorization_layer.out_features)
    )
    mask = torch.full((n_samples, factorization_layer.out_features), True, dtype=torch.bool)
    if num_reps is not None:
        repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    else:
        repetition_index = None
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = factorization_layer.sample(data=data, sampling_ctx=sampling_ctx)
    assert samples.shape == data.shape
    samples_query = samples[:, factorization_layer.scope.query]
    assert torch.isfinite(samples_query).all()


def test_factorization():
    data = make_normal_data(out_features=4)
    factorization = make_product(in_channels=3, out_features=4, num_repetitions=5)
    factorization = expectation_maximization(factorization, data, max_steps=10)
    assert factorization is not None


def test_feature_to_scope_basic():
    """Factorize groups input scopes according to indices."""
    out_features = 4
    num_reps = 1
    leaf = make_normal_leaf(out_features=out_features, out_channels=1, num_repetitions=num_reps)
    module = Factorize(inputs=[leaf], depth=1, num_repetitions=num_reps)

    # Force deterministic grouping: outputs are {0,1} and {2,3}
    indices = torch.zeros(out_features, module.out_features, num_reps)
    indices[0:2, 0, 0] = 1
    indices[2:4, 1, 0] = 1
    module.indices = indices

    feature_scopes = module.feature_to_scope

    assert feature_scopes.shape == (module.out_features, num_reps)
    expected_scope_0 = Scope.join_all([leaf.feature_to_scope[0, 0], leaf.feature_to_scope[1, 0]])
    expected_scope_1 = Scope.join_all([leaf.feature_to_scope[2, 0], leaf.feature_to_scope[3, 0]])
    assert feature_scopes[0, 0] == expected_scope_0
    assert feature_scopes[1, 0] == expected_scope_1
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten())


def test_feature_to_scope_multiple_repetitions():
    """Each repetition uses its own factorization pattern."""
    out_features = 4
    num_reps = 2
    leaf = make_normal_leaf(out_features=out_features, out_channels=1, num_repetitions=num_reps)
    module = Factorize(inputs=[leaf], depth=1, num_repetitions=num_reps)

    # rep 0: {0,1}, {2,3}; rep 1: {0,2}, {1,3}
    indices = torch.zeros(out_features, module.out_features, num_reps)
    indices[0:2, 0, 0] = 1
    indices[2:4, 1, 0] = 1
    indices[[0, 2], 0, 1] = 1
    indices[[1, 3], 1, 1] = 1
    module.indices = indices

    feature_scopes = module.feature_to_scope

    assert feature_scopes.shape == (module.out_features, num_reps)
    expected_rep0 = [
        Scope.join_all([leaf.feature_to_scope[0, 0], leaf.feature_to_scope[1, 0]]),
        Scope.join_all([leaf.feature_to_scope[2, 0], leaf.feature_to_scope[3, 0]]),
    ]
    expected_rep1 = [
        Scope.join_all([leaf.feature_to_scope[0, 1], leaf.feature_to_scope[2, 1]]),
        Scope.join_all([leaf.feature_to_scope[1, 1], leaf.feature_to_scope[3, 1]]),
    ]

    assert feature_scopes[0, 0] == expected_rep0[0]
    assert feature_scopes[1, 0] == expected_rep0[1]
    assert feature_scopes[0, 1] == expected_rep1[0]
    assert feature_scopes[1, 1] == expected_rep1[1]
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten())


@pytest.mark.parametrize(
    "prune,in_channels,marg_rvs,num_reps",
    product(
        [True, False],
        in_channels_values,
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        num_repetitions,
    ),
)
def test_marginalize(prune, in_channels: int, marg_rvs: list[int], num_reps):
    out_features = 6
    module = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)

    # Marginalize scope
    marginalized_module = module.marginalize(marg_rvs, prune=prune)

    if len(marg_rvs) == out_features:
        assert marginalized_module is None
        return

    if prune and len(marg_rvs) == (out_features - 1):
        # If pruning is active and only one scope is left, the (pruned) input module should be returned
        assert isinstance(marginalized_module, type(module.inputs))

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


def test_multidistribution_input():
    out_channels = 3
    num_reps = 5
    out_features_1 = 8
    out_features_2 = 10

    scope_1 = Scope(list(range(0, out_features_1)))
    scope_2 = Scope(list(range(out_features_1, out_features_1 + out_features_2)))

    cls_1 = DummyLeaf
    cls_2 = DummyLeaf

    module_1 = make_leaf(cls=cls_1, out_channels=out_channels, scope=scope_1, num_repetitions=num_reps)
    data_1 = make_data(cls=cls_1, out_features=out_features_1, n_samples=5)

    module_2 = make_leaf(cls=cls_2, out_channels=out_channels, scope=scope_2, num_repetitions=num_reps)
    data_2 = make_data(cls=cls_2, out_features=out_features_2, n_samples=5)

    module = Factorize(inputs=[module_1, module_2], depth=2, num_repetitions=num_reps)

    data = torch.cat((data_1, data_2), dim=1)
    lls = module.log_likelihood(data)

    assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)

    repetition_idx = torch.zeros((1,), dtype=torch.long)
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, num_samples=1)
    sampling_ctx.repetition_idx = repetition_idx

    # Create data tensor to populate with samples
    data_to_sample = torch.full((1, out_features_1 + out_features_2), torch.nan)
    samples = module.sample(data=data_to_sample, sampling_ctx=sampling_ctx)

    assert samples.shape == (1, out_features_1 + out_features_2)


def test_insufficient_features_for_depth():
    """Test that StructureError is raised when depth requires more features than available."""
    # Create a leaves with only 4 features
    out_features = 4
    num_reps = 5
    leaf = make_normal_leaf(out_features=out_features, out_channels=1, num_repetitions=num_reps)

    # Try to create a Factorize with depth=3 (requires 2^3 = 8 features, but only have 4)
    with pytest.raises(StructureError):
        Factorize(inputs=[leaf], depth=3, num_repetitions=num_reps)


def test_exact_feature_count_for_depth():
    """Test that Factorize works when features exactly match required count for depth."""
    # Create a leaves with exactly 8 features
    out_features = 8
    num_reps = 5
    leaf = make_normal_leaf(out_features=out_features, out_channels=1, num_repetitions=num_reps)

    # This should work fine: depth=3 requires 2^3 = 8 features
    factorize = Factorize(inputs=[leaf], depth=3, num_repetitions=num_reps)
    assert factorize.out_features == 8


def test_excess_features_for_depth():
    """Test that Factorize works when features exceed required count for depth."""
    # Create a leaves with 10 features
    out_features = 10
    num_reps = 5
    leaf = make_normal_leaf(out_features=out_features, out_channels=1, num_repetitions=num_reps)

    # This should work fine: depth=2 requires 2^2 = 4 features, and we have 10
    factorize = Factorize(inputs=[leaf], depth=2, num_repetitions=num_reps)
    assert factorize.out_features == 4
