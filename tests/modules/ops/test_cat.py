from itertools import product

import numpy as np
import pytest
import torch

from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.meta import Scope
from spflow.modules.leaves import Categorical, Binomial
from spflow.modules.ops import Cat
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data

out_channels_values = [1, 5]
out_features_values = [1, 6]
dim_values = [1, 2]
num_repetitions = [1, 5]
params = list(product(out_channels_values, out_features_values, num_repetitions, dim_values))


def make_cat(out_channels=3, out_features=3, num_repetitions=None, dim=1):
    if dim == 1:
        # different scopes
        scope_a = Scope(list(range(0, out_features)))
        scope_b = Scope(list(range(out_features, 2 * out_features)))
    elif dim == 2:
        # Same scopes
        scope_a = Scope(list(range(0, out_features)))
        scope_b = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    return Cat(inputs=[inputs_a, inputs_b], dim=dim)


@pytest.mark.parametrize("out_channels,out_features,num_reps, dim", params)
def test_log_likelihood(out_channels: int, out_features: int, num_reps, dim: int, device):
    out_channels = 3
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    ).to(device)
    data = make_normal_data(out_features=module.out_features).to(device)
    lls = module.log_likelihood(data)
    # Always expect 4D output [batch, features, channels, num_reps]
    assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_sample(out_channels: int, out_features: int, num_reps, dim: int, device):
    n_samples = 10
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    ).to(device)
    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan).to(device)
        channel_index = torch.randint(
            low=0, high=module.out_channels, size=(n_samples, module.out_features)
        ).to(device)
        mask = torch.full((n_samples, module.out_features), True, dtype=torch.bool).to(device)
        # Always set repetition_index since num_reps is never None
        repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,)).to(device)
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )
        samples = module.sample(data=data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_expectation_maximization(out_channels: int, out_features: int, num_reps, dim: int, device):
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    ).to(device)
    data = make_normal_data(out_features=module.out_features).to(device)
    expectation_maximization(module, data, max_steps=10)


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_gradient_descent_optimization(out_channels: int, out_features: int, num_reps, dim: int, device):
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    ).to(device)
    data = make_normal_data(out_features=module.out_features).to(device)

    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    train_gradient_descent(module, data_loader, epochs=10)


def test_invalid_constructor_same_scope_dim1(device):
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=1).to(device)


def test_invalid_constructor_different_scope_dim2(device):
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=2).to(device)


def test_invalid_constructor_different_channels_dim1(device):
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels + 1, num_repetitions=num_repetitions).to(
        device
    )

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=1).to(device)


def test_invalid_constructor_different_features_dim2(device):
    out_features = 3
    out_channels = 3
    num_repetitions = 3

    inputs_a = make_normal_leaf(
        out_features=out_features, out_channels=out_channels, num_repetitions=num_repetitions
    ).to(device)
    inputs_b = make_normal_leaf(
        out_features=out_features + 1, out_channels=out_channels, num_repetitions=num_repetitions
    ).to(device)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=2).to(device)


@pytest.mark.parametrize(
    "prune,out_channels,dim,marg_rvs,num_reps",
    product(
        [True, False],
        out_channels_values,
        dim_values,
        # Create a list of all combinations betweens elements of [{}, 0, 1, 2, 3] and [{}, 0, 1, 2, 3] (where {} means not present
        [
            [0],
            [1],
            [2],
            [3],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
        ],
        num_repetitions,
    ),
)
def test_marginalize(prune, out_channels: int, dim: int, marg_rvs: list[int], num_reps, device):
    out_features = 4
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        dim=dim,
        num_repetitions=num_reps,
    ).to(device)

    # Marginalize scope
    marginalized_module = module.marginalize(marg_rvs, prune=prune)

    if len(marg_rvs) == module.out_features:
        assert marginalized_module is None
        return
    else:
        assert marginalized_module.out_features == module.out_features - len(marg_rvs)

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


def test_marginalize_one_of_two_inputs(device):
    out_channels = 3
    num_repetitions = 3
    inputs_cat = Categorical(
        scope=Scope([0, 1]),
        probs=torch.rand(2, out_channels, num_repetitions),
        num_repetitions=num_repetitions,
    ).to(device)
    inputs_bin = Binomial(
        scope=Scope([2, 3, 4]),
        num_repetitions=num_repetitions,
        total_count=torch.ones((3, out_channels, num_repetitions)) * 3,
        probs=torch.rand(3, out_channels, num_repetitions),
    ).to(device)

    module = Cat(inputs=[inputs_cat, inputs_bin], dim=1).to(device)

    # Marginalize categorical scope, expect binomial to be returned
    marg_rvs_cat = inputs_cat.scope.query
    marginalized_module = module.marginalize(marg_rvs_cat, prune=True)
    assert isinstance(marginalized_module, type(inputs_bin))

    # Marginalize binomial scope, expect categorical to be returned
    marg_rvs_bin = inputs_bin.scope.query
    marginalized_module = module.marginalize(marg_rvs_bin, prune=True)
    assert isinstance(marginalized_module, type(inputs_cat))


# Tests for feature_to_scope property


def test_cat_feature_to_scope_dim1_basic():
    """Test feature_to_scope concatenation when dim=1."""
    out_channels = 3
    num_repetitions = 2
    out_features_a = 3
    out_features_b = 4

    # Create two inputs with disjoint scopes
    scope_a = Scope(list(range(0, out_features_a)))
    scope_b = Scope(list(range(out_features_a, out_features_a + out_features_b)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    cat = Cat(inputs=[inputs_a, inputs_b], dim=1)

    # Get feature_to_scope arrays
    feature_scopes = cat.feature_to_scope
    input_a_scopes = inputs_a.feature_to_scope
    input_b_scopes = inputs_b.feature_to_scope

    # Should concatenate along axis 0 (features dimension)
    expected_shape = (out_features_a + out_features_b, num_repetitions)
    assert feature_scopes.shape == expected_shape

    # Verify concatenation is correct
    assert np.array_equal(feature_scopes[:out_features_a], input_a_scopes)
    assert np.array_equal(feature_scopes[out_features_a:], input_b_scopes)

    # All elements should be Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_cat_feature_to_scope_dim1_multiple_inputs():
    """Test feature_to_scope with more than 2 inputs when dim=1."""
    out_channels = 2
    num_repetitions = 3
    out_features_per_input = [2, 3, 4]

    # Create three inputs with disjoint scopes
    inputs = []
    start = 0
    for out_features in out_features_per_input:
        scope = Scope(list(range(start, start + out_features)))
        inputs.append(make_normal_leaf(scope, out_channels=out_channels, num_repetitions=num_repetitions))
        start += out_features

    cat = Cat(inputs=inputs, dim=1)

    # Get feature_to_scope
    feature_scopes = cat.feature_to_scope

    # Verify shape
    total_features = sum(out_features_per_input)
    assert feature_scopes.shape == (total_features, num_repetitions)

    # Verify concatenation order
    offset = 0
    for i, inp in enumerate(inputs):
        inp_scopes = inp.feature_to_scope
        out_features = out_features_per_input[i]
        assert np.array_equal(feature_scopes[offset : offset + out_features], inp_scopes)
        offset += out_features

    # All elements should be Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_cat_feature_to_scope_dim1_repetitions():
    """Test feature_to_scope with multiple repetitions when dim=1."""
    out_channels = 4
    num_repetitions = 5
    out_features_a = 2
    out_features_b = 3

    scope_a = Scope(list(range(0, out_features_a)))
    scope_b = Scope(list(range(out_features_a, out_features_a + out_features_b)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    cat = Cat(inputs=[inputs_a, inputs_b], dim=1)

    feature_scopes = cat.feature_to_scope

    # Verify shape includes all repetitions
    assert feature_scopes.shape == (out_features_a + out_features_b, num_repetitions)

    # Verify each repetition column is consistent
    for rep_idx in range(num_repetitions):
        # First out_features_a rows should match input_a's scopes for this repetition
        for feat_idx in range(out_features_a):
            expected_scope = Scope([scope_a.query[feat_idx]])
            assert feature_scopes[feat_idx, rep_idx] == expected_scope

        # Next out_features_b rows should match input_b's scopes for this repetition
        for feat_idx in range(out_features_b):
            expected_scope = Scope([scope_b.query[feat_idx]])
            assert feature_scopes[out_features_a + feat_idx, rep_idx] == expected_scope

    # All elements should be Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_cat_feature_to_scope_dim2_passthrough():
    """Test feature_to_scope uses first input when dim=2."""
    out_features = 4
    out_channels_per_input = 2
    num_repetitions = 3

    scope = Scope(list(range(0, out_features)))

    # Create two inputs with same scope (required for dim=2)
    inputs_a = make_normal_leaf(scope, out_channels=out_channels_per_input, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope, out_channels=out_channels_per_input, num_repetitions=num_repetitions)

    cat = Cat(inputs=[inputs_a, inputs_b], dim=2)

    # Get feature_to_scope arrays
    feature_scopes = cat.feature_to_scope
    input_a_scopes = inputs_a.feature_to_scope

    # Should be identical to first input (pass-through)
    assert np.array_equal(feature_scopes, input_a_scopes)
    assert feature_scopes.shape == (out_features, num_repetitions)

    # All elements should be Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_cat_feature_to_scope_dim2_multiple_inputs():
    """Test feature_to_scope pass-through with multiple inputs when dim=2."""
    out_features = 3
    out_channels = 2
    num_repetitions = 4

    scope = Scope(list(range(0, out_features)))

    # Create three inputs with same scope
    inputs = [
        make_normal_leaf(scope, out_channels=out_channels, num_repetitions=num_repetitions) for _ in range(3)
    ]

    cat = Cat(inputs=inputs, dim=2)

    feature_scopes = cat.feature_to_scope
    first_input_scopes = inputs[0].feature_to_scope

    # Should match first input
    assert np.array_equal(feature_scopes, first_input_scopes)
    assert feature_scopes.shape == (out_features, num_repetitions)

    # All elements should be Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_cat_feature_to_scope_scope_correctness_dim1():
    """Test that individual Scope objects are correct when dim=1."""
    out_channels = 2
    num_repetitions = 2
    out_features_a = 2
    out_features_b = 3

    scope_a = Scope([10, 20])  # Non-contiguous RVs
    scope_b = Scope([30, 40, 50])

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    cat = Cat(inputs=[inputs_a, inputs_b], dim=1)

    feature_scopes = cat.feature_to_scope

    # Verify each scope object contains the correct RV
    expected_rvs = [10, 20, 30, 40, 50]
    for feat_idx, expected_rv in enumerate(expected_rvs):
        for rep_idx in range(num_repetitions):
            scope = feature_scopes[feat_idx, rep_idx]
            assert scope == Scope([expected_rv])
            assert scope.query == (expected_rv,)


def test_cat_feature_to_scope_single_repetition():
    """Test feature_to_scope with single repetition (edge case)."""
    out_channels = 3
    num_repetitions = 1
    out_features_a = 3
    out_features_b = 2

    scope_a = Scope(list(range(0, out_features_a)))
    scope_b = Scope(list(range(out_features_a, out_features_a + out_features_b)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    cat = Cat(inputs=[inputs_a, inputs_b], dim=1)

    feature_scopes = cat.feature_to_scope

    # Should still be 2D with shape (total_features, 1)
    assert feature_scopes.shape == (out_features_a + out_features_b, 1)

    # All elements should be Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


@pytest.mark.parametrize("num_repetitions", [1, 3, 7])
def test_cat_feature_to_scope_dim1_various_repetitions(num_repetitions):
    """Test feature_to_scope concatenation with various repetition counts."""
    out_channels = 2
    out_features_a = 3
    out_features_b = 4

    scope_a = Scope(list(range(0, out_features_a)))
    scope_b = Scope(list(range(out_features_a, out_features_a + out_features_b)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    cat = Cat(inputs=[inputs_a, inputs_b], dim=1)

    feature_scopes = cat.feature_to_scope

    # Verify shape
    assert feature_scopes.shape == (out_features_a + out_features_b, num_repetitions)

    # Verify concatenation is correct for all repetitions
    input_a_scopes = inputs_a.feature_to_scope
    input_b_scopes = inputs_b.feature_to_scope

    assert np.array_equal(feature_scopes[:out_features_a], input_a_scopes)
    assert np.array_equal(feature_scopes[out_features_a:], input_b_scopes)

    # All elements should be Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())
