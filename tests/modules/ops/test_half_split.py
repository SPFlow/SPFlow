from itertools import product

import numpy as np
import pytest
import torch

from spflow.meta import Scope
from spflow.modules.ops import SplitConsecutive
from spflow.modules.products import ElementwiseProduct, OuterProduct
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import make_normal_leaf, make_normal_data
from tests.utils.sampling_context_helpers import assert_nonzero_finite_grad, make_diff_routing_from_logits

cls = [ElementwiseProduct, OuterProduct]

out_channels_values = [1, 4]
out_features_values = [2, 4]
num_repetition_values = [1, 5]


@pytest.mark.parametrize(
    "cls,out_channels,out_features,num_repetitions",
    product(cls, out_channels_values, out_features_values, num_repetition_values),
)
def test_split_result(cls, out_channels: int, out_features: int, num_repetitions: int):
    out_channels = out_channels
    num_features = out_features
    scope = Scope(list(range(0, num_features)))
    scope_1 = Scope(list(range(0, num_features // 2)))
    scope_2 = Scope(list(range(num_features // 2, num_features)))
    mean = torch.randn(num_features, out_channels, num_repetitions)
    std = torch.rand(num_features, out_channels, num_repetitions)
    leaf = make_normal_leaf(scope=scope, mean=mean, std=std)
    leaf_half_1 = make_normal_leaf(
        scope=scope_1, mean=mean[: num_features // 2], std=std[: num_features // 2]
    )
    leaf_half_2 = make_normal_leaf(
        scope=scope_2, mean=mean[num_features // 2 :], std=std[num_features // 2 :]
    )
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1)
    spn1 = cls(inputs=split)
    spn2 = cls(inputs=[leaf_half_1, leaf_half_2])
    assert spn1.out_shape.channels == spn2.out_shape.channels
    assert spn1.out_shape.features == spn2.out_shape.features
    data = make_normal_data(out_features=num_features)
    ll_1 = spn1.log_likelihood(data)
    ll_2 = spn2.log_likelihood(data)
    torch.testing.assert_close(ll_1, ll_2, rtol=1e-5, atol=1e-6)

    n_samples = 100
    num_inputs = 2

    data1 = torch.full((n_samples, spn1.out_shape.features * num_inputs), torch.nan)
    data2 = torch.full((n_samples, spn1.out_shape.features * num_inputs), torch.nan)
    mask = torch.full((n_samples, spn1.out_shape.features), True, dtype=torch.bool)
    channel_index = torch.randint(
        low=0, high=spn1.out_shape.channels, size=(n_samples, spn1.out_shape.features)
    )
    rep_index = torch.randint(low=0, high=num_repetitions, size=(n_samples,))
    sampling_ctx = SamplingContext(
        channel_index=channel_index, repetition_index=rep_index, mask=mask, is_mpe=True
    )
    sampling_ctx2 = SamplingContext(
        channel_index=channel_index, repetition_index=rep_index, mask=mask, is_mpe=True
    )
    s1 = spn1._sample(data=data1, sampling_ctx=sampling_ctx, cache=Cache())
    s2 = spn2._sample(data=data2, sampling_ctx=sampling_ctx2, cache=Cache())

    torch.testing.assert_close(s1, s2, rtol=0.0, atol=0.0)


# New tests for Phase 3 coverage improvement


def test_split_mode_extra_repr():
    """Test string representation of SplitConsecutive."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1)

    repr_str = split.extra_repr()
    assert isinstance(repr_str, str)
    assert "dim=1" in repr_str


def test_split_mode_feature_to_scope():
    """Test feature_to_scope property delegates to input."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1)

    # Split operations delegate to input's feature_to_scope
    feature_scopes = split.feature_to_scope

    # Should be identical to the input's feature_to_scope
    assert feature_scopes.shape == (3, 2, 1)
    assert np.array_equal(
        feature_scopes,
        np.array([[Scope(0), Scope(1), Scope(2)], [Scope(3), Scope(4), Scope(5)]]).reshape(3, 2, 1),
    )

    # Each element should be a Scope object
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


@pytest.mark.parametrize("num_features,num_splits", [(6, 2), (9, 3), (12, 3)])
def test_split_mode_uneven_features(num_features, num_splits):
    """Test with features that divide evenly (testing behavior is correct)."""
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitConsecutive(inputs=leaf, num_splits=num_splits, dim=1)

    # Test that splitting still works
    data = make_normal_data(out_features=num_features)
    lls = split.log_likelihood(data)

    # torch.split can create more chunks if size doesn't divide evenly
    # This is expected behavior
    assert len(lls) >= num_splits
    # Verify all log likelihoods have valid shapes
    for ll in lls:
        assert ll.ndim == 4
        assert ll.shape[0] == data.shape[0]  # batch size


def test_split_mode_single_feature():
    """Test with single feature (edge case)."""
    scope = Scope([0])
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitConsecutive(inputs=leaf, num_splits=1, dim=1)

    data = make_normal_data(out_features=1)
    lls = split.log_likelihood(data)

    assert len(lls) == 1
    assert lls[0].shape == (data.shape[0], 1, 2, 1)


def test_split_mode_many_features():
    """Test with many features."""
    num_features = 20
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2)
    split = SplitConsecutive(inputs=leaf, num_splits=4, dim=1)

    data = make_normal_data(out_features=num_features)
    lls = split.log_likelihood(data)

    assert len(lls) == 4
    for ll in lls:
        assert ll.shape[0] == data.shape[0]
        assert ll.shape[1] == num_features // 4


def test_split_mode_log_likelihood_consistency():
    """Test log_likelihood produces consistent results."""
    scope = Scope(list(range(0, 10)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1)

    data = make_normal_data(out_features=10)
    lls1 = split.log_likelihood(data)
    lls2 = split.log_likelihood(data)

    # Results should be identical for same input
    assert len(lls1) == len(lls2)
    for ll1, ll2 in zip(lls1, lls2):
        torch.testing.assert_close(ll1, ll2, rtol=0.0, atol=0.0)


def test_split_mode_sampling_consistency():
    """Test sampling produces valid samples."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1)

    n_samples = 20
    data = torch.full((n_samples, 6), torch.nan)
    channel_index = torch.randint(0, 3, size=(n_samples, 6))
    mask = torch.ones((n_samples, 6), dtype=torch.bool)
    rep_index = torch.randint(0, 2, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=rep_index)
    samples = split._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert samples.shape == (n_samples, 6)
    assert torch.isfinite(samples).all()


def test_split_mode_differentiable_equals_non_diff_sampling():
    scope = Scope(list(range(0, 6)))
    num_reps = 3
    split = SplitConsecutive(
        inputs=make_normal_leaf(scope, out_channels=3, num_repetitions=num_reps),
        num_splits=2,
        dim=1,
    )
    n_samples = 14
    channel_index = torch.randint(0, split.out_shape.channels, size=(n_samples, split.out_shape.features))
    mask = torch.ones((n_samples, split.out_shape.features), dtype=torch.bool)
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index.clone(),
        mask=mask.clone(),
        repetition_index=repetition_index.clone(),
    )
    sampling_ctx_b = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=split.out_shape.channels),
        mask=mask.clone(),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_differentiable=True,
        hard=True,
    )

    torch.manual_seed(1337)
    samples_a = split._sample(
        data=torch.full((n_samples, 6), torch.nan),
        sampling_ctx=sampling_ctx_a,
        cache=Cache(),
    )
    torch.manual_seed(1337)
    samples_b = split._sample(
        data=torch.full((n_samples, 6), torch.nan),
        sampling_ctx=sampling_ctx_b,
        cache=Cache(),
    )

    torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)


def test_split_mode_differentiable_gradients_flow():
    scope = Scope(list(range(0, 6)))
    num_reps = 3
    split = SplitConsecutive(
        inputs=make_normal_leaf(scope, out_channels=3, num_repetitions=num_reps),
        num_splits=2,
        dim=1,
    )
    n_samples = 10
    channel_logits, repetition_logits, channel_index, repetition_index = make_diff_routing_from_logits(
        num_samples=n_samples,
        num_features=split.out_shape.features,
        num_channels=split.out_shape.channels,
        num_repetitions=num_reps,
    )
    sampling_ctx = SamplingContext(
        channel_index=channel_index,
        mask=torch.ones((n_samples, split.out_shape.features), dtype=torch.bool),
        repetition_index=repetition_index,
        is_differentiable=True,
        hard=False,
    )
    out = split._sample(
        data=torch.full((n_samples, 6), torch.nan),
        sampling_ctx=sampling_ctx,
        cache=Cache(),
    )
    loss = torch.nan_to_num(out).sum()
    loss.backward()

    assert_nonzero_finite_grad(channel_logits, "channel_logits")
    assert_nonzero_finite_grad(repetition_logits, "repetition_logits")


@pytest.mark.parametrize("dim", [1, 2])
def test_split_mode_different_dims(dim):
    """Test splitting along different dimensions."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=4, num_repetitions=1)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=dim)

    data = make_normal_data(out_features=6)
    lls = split.log_likelihood(data)

    assert len(lls) == 2
    for ll in lls:
        assert torch.isfinite(ll).all()
