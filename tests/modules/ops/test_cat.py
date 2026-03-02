from itertools import product

import numpy as np
import pytest
import torch

from spflow.exceptions import InvalidParameterError
from spflow.meta import Scope
from spflow.modules.leaves import Binomial, Categorical, Normal
from spflow.modules.ops import Cat
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import make_normal_leaf
from tests.utils.sampling_context_helpers import make_sampling_context

out_channels_values = [1, 5]
dim_values = [1, 2]
num_repetitions = [1, 5]


def _rand(*size: int) -> torch.Tensor:
    return torch.rand(*size)


def make_cat(out_channels=3, out_features=3, num_repetitions=None, dim=1):
    if dim == 1:
        # dim=1 concatenates features, so child scopes must stay disjoint.
        scope_a = Scope(list(range(0, out_features)))
        scope_b = Scope(list(range(out_features, 2 * out_features)))
    elif dim == 2:
        # dim=2 concatenates channels, so every child must cover the same RVs.
        scope_a = Scope(list(range(0, out_features)))
        scope_b = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    return Cat(inputs=[inputs_a, inputs_b], dim=dim)


# Cross-module Cat contracts moved to:
# - test_ops_cat_contract.py


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
    sampling_ctx = make_sampling_context(
        num_samples=num_samples,
        num_features=2,
        num_channels=4,
        num_repetitions=1,
        channel_index=torch.zeros((num_samples, 2), dtype=torch.long),
        mask=torch.ones((num_samples, 2), dtype=torch.bool),
        repetition_index=torch.zeros((num_samples,), dtype=torch.long),
    )
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
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
    # Fail-fast protects against silently clipping invalid routing decisions.
    with pytest.raises(InvalidParameterError, match="out-of-range channel ids"):
        module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())


def test_sample_dim2_differentiable_routes_child_offsets():
    scope = Scope([0, 1])
    num_samples = 5
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

    channel_ids = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [1, 3]], dtype=torch.long)
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(channel_ids, dim=-1, dim_size=module.out_shape.channels),
        mask=torch.ones((num_samples, 2), dtype=torch.bool),
        repetition_index=to_one_hot(torch.zeros((num_samples,), dtype=torch.long), dim=-1, dim_size=1),
        is_differentiable=True,
    )
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == (num_samples, 2)
    assert torch.isfinite(samples).all()
    expected = torch.where(
        channel_ids == 0, channel_ids.new_tensor(100.0), (channel_ids - 1).to(samples.dtype) * 10.0
    )
    torch.testing.assert_close(samples, expected.to(samples.dtype), rtol=0.0, atol=1e-3)


def test_sample_dim2_differentiable_equals_non_diff_sampling():
    scope = Scope([0, 1])
    num_samples = 6
    child_a = Normal(
        scope=scope,
        out_channels=2,
        num_repetitions=1,
        loc=torch.tensor([100.0, 200.0], dtype=torch.get_default_dtype()).view(1, 2, 1).repeat(2, 1, 1),
        scale=torch.full((2, 2, 1), 1e-6),
    )
    child_b = Normal(
        scope=scope,
        out_channels=3,
        num_repetitions=1,
        loc=torch.tensor([0.0, 10.0, 20.0], dtype=torch.get_default_dtype()).view(1, 3, 1).repeat(2, 1, 1),
        scale=torch.full((2, 3, 1), 1e-6),
    )
    module = Cat(inputs=[child_a, child_b], dim=2)
    channel_ids = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [1, 4]], dtype=torch.long)
    mask = torch.ones((num_samples, 2), dtype=torch.bool)
    repetition_ids = torch.zeros((num_samples,), dtype=torch.long)

    sampling_ctx_a = SamplingContext(
        channel_index=channel_ids.clone(),
        mask=mask.clone(),
        repetition_index=repetition_ids.clone(),
    )
    sampling_ctx_b = SamplingContext(
        channel_index=to_one_hot(channel_ids, dim=-1, dim_size=module.out_shape.channels),
        mask=mask.clone(),
        repetition_index=to_one_hot(repetition_ids, dim=-1, dim_size=1),
        is_differentiable=True,
    )

    samples_a = module._sample(
        data=torch.full((num_samples, 2), torch.nan),
        sampling_ctx=sampling_ctx_a,
        cache=Cache(),
    )
    samples_b = module._sample(
        data=torch.full((num_samples, 2), torch.nan),
        sampling_ctx=sampling_ctx_b,
        cache=Cache(),
    )

    torch.testing.assert_close(samples_a, samples_b, rtol=0.0, atol=1e-4)
    torch.testing.assert_close(
        sampling_ctx_b.channel_index,
        to_one_hot(sampling_ctx_a.channel_index, dim=-1, dim_size=module.out_shape.channels),
        rtol=0.0,
        atol=0.0,
    )


def test_sample_dim2_differentiable_multi_child_routing_rejected():
    scope = Scope([0, 1])
    num_samples = 2
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

    channel_index = torch.zeros((num_samples, 2, module.out_shape.channels), dtype=torch.get_default_dtype())
    channel_index[0, 0, 0] = 1.0
    channel_index[0, 0, 1] = 1.0
    channel_index[0, 1, 0] = 1.0
    channel_index[1, 0, 0] = 1.0
    channel_index[1, 1, 1] = 1.0
    sampling_ctx = SamplingContext(
        channel_index=channel_index,
        mask=torch.ones((num_samples, 2), dtype=torch.bool),
        repetition_index=to_one_hot(torch.zeros((num_samples,), dtype=torch.long), dim=-1, dim_size=1),
        is_differentiable=True,
    )

    with pytest.raises(InvalidParameterError, match="select exactly one child"):
        module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())


def test_sample_does_not_mutate_parent_sampling_context():
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
        out_channels=2,
        num_repetitions=1,
        loc=torch.tensor([0.0, 10.0], dtype=torch.get_default_dtype()).view(1, 2, 1).repeat(2, 1, 1),
        scale=torch.full((2, 2, 1), 1e-6),
    )
    module = Cat(inputs=[child_a, child_b], dim=2)
    data = torch.full((3, 2), torch.nan)
    sampling_ctx = make_sampling_context(
        num_samples=3,
        num_features=2,
        num_channels=3,
        num_repetitions=1,
        channel_index=torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long),
        mask=torch.tensor([[True, True], [True, False], [False, True]], dtype=torch.bool),
    )
    channel_before = sampling_ctx.channel_index.clone()
    mask_before = sampling_ctx.mask.clone()

    module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    # Parent context is shared across traversals, so Cat must treat it as immutable input.
    assert torch.equal(sampling_ctx.channel_index, channel_before)
    assert torch.equal(sampling_ctx.mask, mask_before)


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
        # Keep a broad set to stress interactions between removed RV subsets.
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

    # This guards the invariant that marginalized RVs disappear from output scope.
    marginalized_module = module.marginalize(marg_rvs, prune=prune)

    if len(marg_rvs) == module.out_shape.features:
        assert marginalized_module is None
        return
    else:
        assert marginalized_module.out_shape.features == module.out_shape.features - len(marg_rvs)

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

    # Pruning should collapse Cat to the surviving branch, not keep an extra wrapper.
    marg_rvs_cat = inputs_cat.scope.query
    marginalized_module = module.marginalize(marg_rvs_cat, prune=True)
    assert isinstance(marginalized_module, type(inputs_bin))

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

    scope_a = Scope(list(range(0, out_features_a)))
    scope_b = Scope(list(range(out_features_a, out_features_a + out_features_b)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    cat = Cat(inputs=[inputs_a, inputs_b], dim=1)

    feature_scopes = cat.feature_to_scope
    input_a_scopes = inputs_a.feature_to_scope
    input_b_scopes = inputs_b.feature_to_scope

    # feature_to_scope must track the same feature ordering Cat exposes downstream.
    expected_shape = (out_features_a + out_features_b, num_repetitions)
    assert feature_scopes.shape == expected_shape

    assert np.array_equal(feature_scopes[:out_features_a], input_a_scopes)
    assert np.array_equal(feature_scopes[out_features_a:], input_b_scopes)

    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_cat_feature_to_scope_dim1_multiple_inputs():
    """Test feature_to_scope with more than 2 inputs when dim=1."""
    out_channels = 2
    num_repetitions = 3
    out_features_per_input = [2, 3, 4]

    inputs = []
    start = 0
    for out_features in out_features_per_input:
        scope = Scope(list(range(start, start + out_features)))
        inputs.append(make_normal_leaf(scope, out_channels=out_channels, num_repetitions=num_repetitions))
        start += out_features

    cat = Cat(inputs=inputs, dim=1)

    feature_scopes = cat.feature_to_scope

    total_features = sum(out_features_per_input)
    assert feature_scopes.shape == (total_features, num_repetitions)

    # Order checks prevent silent reindexing bugs when Cat chains many children.
    offset = 0
    for i, inp in enumerate(inputs):
        inp_scopes = inp.feature_to_scope
        out_features = out_features_per_input[i]
        assert np.array_equal(feature_scopes[offset : offset + out_features], inp_scopes)
        offset += out_features

    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_cat_feature_to_scope_dim2_passthrough():
    """Test feature_to_scope uses first input when dim=2."""
    out_features = 4
    out_channels_per_input = 2
    num_repetitions = 3

    scope = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope, out_channels=out_channels_per_input, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope, out_channels=out_channels_per_input, num_repetitions=num_repetitions)

    cat = Cat(inputs=[inputs_a, inputs_b], dim=2)

    feature_scopes = cat.feature_to_scope
    input_a_scopes = inputs_a.feature_to_scope

    # dim=2 should preserve feature mapping exactly; channels change, RV order must not.
    assert np.array_equal(feature_scopes, input_a_scopes)
    assert feature_scopes.shape == (out_features, num_repetitions)

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

    # Non-contiguous RV ids catch accidental assumptions about contiguous indexing.
    expected_rvs = [10, 20, 30, 40, 50]
    for feat_idx, expected_rv in enumerate(expected_rvs):
        for rep_idx in range(num_repetitions):
            scope = feature_scopes[feat_idx, rep_idx]
            assert scope == Scope([expected_rv])
            assert scope.query == (expected_rv,)


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

    assert feature_scopes.shape == (out_features_a + out_features_b, num_repetitions)

    # Repetition axis must not alter scope layout semantics.
    input_a_scopes = inputs_a.feature_to_scope
    input_b_scopes = inputs_b.feature_to_scope

    assert np.array_equal(feature_scopes[:out_features_a], input_a_scopes)
    assert np.array_equal(feature_scopes[out_features_a:], input_b_scopes)

    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())
