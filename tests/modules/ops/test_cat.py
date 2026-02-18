from itertools import product

import numpy as np
import pytest
import torch

from spflow.exceptions import InvalidParameterError
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.meta import Scope
from spflow.modules.leaves import Binomial, Categorical, Normal
from spflow.modules.ops import Cat
from spflow.utils.cache import Cache
from tests.utils.leaves import make_normal_leaf, make_normal_data
from tests.utils.sampling_context_helpers import make_sampling_context

out_channels_values = [1, 5]
out_features_values = [1, 6]
dim_values = [1, 2]
num_repetitions = [1, 5]
params = list(product(out_channels_values, out_features_values, num_repetitions, dim_values))


def _randint(low: int, high: int, size: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(low=low, high=high, size=size)


def _rand(*size: int) -> torch.Tensor:
    return torch.rand(*size)


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
def test_log_likelihood(out_channels: int, out_features: int, num_reps, dim: int):
    out_channels = 3
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = make_normal_data(out_features=module.out_shape.features)
    lls = module.log_likelihood(data)
    # Always expect 4D output [batch, features, channels, num_reps]
    assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_sample(out_channels: int, out_features: int, num_reps, dim: int):
    n_samples = 10
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = torch.full((n_samples, module.out_shape.features), torch.nan)
    samples = module.sample(data=data)
    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


def test_sample_dim2_defaults_to_first_global_channel():
    scope = Scope([0, 1])
    num_samples = 4

    # Child A owns global channel [0], Child B owns global channels [1, 2, 3].
    child_a = Normal(
        scope=scope,
        out_channels=1,
        num_repetitions=1,
        loc=torch.full((2, 1, 1), 100.0),
        scale=torch.full((2, 1, 1), 1e-6),
    )
    child_b = Normal(
        scope=scope,
        out_channels=3,
        num_repetitions=1,
        loc=torch.tensor([0.0, 10.0, 20.0], dtype=torch.get_default_dtype()).view(1, 3, 1).repeat(2, 1, 1),
        scale=torch.full((2, 3, 1), 1e-6),
    )
    module = Cat(inputs=[child_a, child_b], dim=2)

    data = torch.full((num_samples, 2), torch.nan)

    samples = module.sample(data=data)
    expected = torch.full((2,), 100.0, dtype=samples.dtype)
    for row_idx in range(num_samples):
        torch.testing.assert_close(
            samples[row_idx],
            expected,
            rtol=0.0,
            atol=1e-4,
        )


def test_sample_dim2_routes_unequal_child_channels_by_offsets_internal_context():
    scope = Scope([0, 1])
    num_samples = 4

    # Child A owns global channel [0], Child B owns global channels [1, 2, 3].
    child_a = Normal(
        scope=scope,
        out_channels=1,
        num_repetitions=1,
        loc=torch.full((2, 1, 1), 100.0),
        scale=torch.full((2, 1, 1), 1e-6),
    )
    child_b = Normal(
        scope=scope,
        out_channels=3,
        num_repetitions=1,
        loc=torch.tensor([0.0, 10.0, 20.0], dtype=torch.get_default_dtype()).view(1, 3, 1).repeat(2, 1, 1),
        scale=torch.full((2, 3, 1), 1e-6),
    )
    module = Cat(inputs=[child_a, child_b], dim=2)

    data = torch.full((num_samples, 2), torch.nan)
    channel_index = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long)
    sampling_ctx = make_sampling_context(
        num_samples=num_samples,
        num_features=2,
        num_channels=4,
        num_repetitions=1,
        channel_index=channel_index,
        mask=torch.ones((num_samples, 2), dtype=torch.bool),
    )

    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    expected = torch.tensor([100.0, 0.0, 10.0, 20.0], dtype=samples.dtype)
    for row_idx, value in enumerate(expected):
        torch.testing.assert_close(
            samples[row_idx],
            torch.full((2,), value.item(), dtype=samples.dtype),
            rtol=0.0,
            atol=1e-4,
        )


def test_sample_dim2_rejects_out_of_range_global_channel_id_internal_context():
    scope = Scope([0, 1])
    child_a = Normal(
        scope=scope,
        out_channels=1,
        num_repetitions=1,
        loc=torch.full((2, 1, 1), 100.0),
        scale=torch.full((2, 1, 1), 1e-6),
    )
    child_b = Normal(
        scope=scope,
        out_channels=3,
        num_repetitions=1,
        loc=torch.tensor([0.0, 10.0, 20.0], dtype=torch.get_default_dtype()).view(1, 3, 1).repeat(2, 1, 1),
        scale=torch.full((2, 3, 1), 1e-6),
    )
    module = Cat(inputs=[child_a, child_b], dim=2)
    data = torch.full((2, 2), torch.nan)

    sampling_ctx = make_sampling_context(
        num_samples=2,
        num_features=2,
        num_channels=5,
        num_repetitions=1,
        channel_index=torch.tensor([[0, 0], [4, 4]], dtype=torch.long),
        mask=torch.ones((2, 2), dtype=torch.bool),
    )
    with pytest.raises(InvalidParameterError, match="out-of-range channel ids"):
        module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_expectation_maximization(out_channels: int, out_features: int, num_reps, dim: int):
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = make_normal_data(out_features=module.out_shape.features)
    locs_before = [inp.loc.detach().clone() for inp in module.inputs]
    scales_before = [inp.scale.detach().clone() for inp in module.inputs]

    max_steps = 2
    ll_history = expectation_maximization(module, data, max_steps=max_steps)
    assert ll_history.ndim == 1
    assert 1 <= ll_history.numel() <= max_steps
    assert ll_history.isfinite().all()

    for i, inp in enumerate(module.inputs):
        assert not torch.equal(inp.loc, locs_before[i])
        assert not torch.equal(inp.scale, scales_before[i])
        torch.testing.assert_close(inp.loc, torch.zeros_like(inp.loc))
        torch.testing.assert_close(inp.scale, torch.ones_like(inp.scale))


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_gradient_descent_optimization(out_channels: int, out_features: int, num_reps, dim: int):
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = make_normal_data(out_features=module.out_shape.features)

    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    loc_before = module.inputs[0].loc.detach().clone()
    log_scale_before = module.inputs[0].log_scale.detach().clone()

    train_gradient_descent(module, data_loader, epochs=1)

    assert not torch.equal(module.inputs[0].loc, loc_before)
    assert not torch.equal(module.inputs[0].log_scale, log_scale_before)


def test_invalid_constructor_same_scope_dim1():
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=1)


def test_invalid_constructor_different_scope_dim2():
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=2)


def test_invalid_constructor_different_channels_dim1():
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels + 1, num_repetitions=num_repetitions)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=1)


def test_invalid_constructor_different_features_dim2():
    out_features = 3
    out_channels = 3
    num_repetitions = 3

    inputs_a = make_normal_leaf(
        out_features=out_features, out_channels=out_channels, num_repetitions=num_repetitions
    )
    inputs_b = make_normal_leaf(
        out_features=out_features + 1, out_channels=out_channels, num_repetitions=num_repetitions
    )

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=2)


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
def test_marginalize(prune, out_channels: int, dim: int, marg_rvs: list[int], num_reps):
    out_features = 4
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        dim=dim,
        num_repetitions=num_reps,
    )

    # Marginalize scope
    marginalized_module = module.marginalize(marg_rvs, prune=prune)

    if len(marg_rvs) == module.out_shape.features:
        assert marginalized_module is None
        return
    else:
        assert marginalized_module.out_shape.features == module.out_shape.features - len(marg_rvs)

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


def test_marginalize_one_of_two_inputs():
    out_channels = 3
    num_repetitions = 3
    inputs_cat = Categorical(
        scope=Scope([0, 1]),
        probs=_rand(2, out_channels, num_repetitions),
        num_repetitions=num_repetitions,
    )
    inputs_bin = Binomial(
        scope=Scope([2, 3, 4]),
        num_repetitions=num_repetitions,
        total_count=torch.ones((3, out_channels, num_repetitions)) * 3,
        probs=_rand(3, out_channels, num_repetitions),
    )

    module = Cat(inputs=[inputs_cat, inputs_bin], dim=1)

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
