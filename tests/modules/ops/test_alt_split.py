from itertools import product

import numpy as np
import pytest
import torch

from spflow.meta import Scope
from spflow.modules.ops import SplitInterleaved
from spflow.modules.ops import SplitConsecutive
from spflow.modules.products import ElementwiseProduct
from spflow.modules.products import OuterProduct
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import DifferentiableSamplingContext, SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data

out_channels_values = [1, 6]
features_values_multiplier = [1, 6]
num_splits = [2, 3]
split_type = [SplitConsecutive, SplitInterleaved]
params = list(product(out_channels_values, features_values_multiplier, num_splits, split_type))


cls = [ElementwiseProduct, OuterProduct]

out_channels_values = [1, 3]
out_features_values = [2, 4]
num_repetition_values = [1, 5]


@pytest.mark.parametrize(
    "cls,out_channels,out_features,num_repetitions",
    product(cls, out_channels_values, out_features_values, num_repetition_values),
)
def test_split_result(cls, out_channels: int, out_features: int, num_repetitions: int):
    out_channels = 10
    num_features = 6
    scope = Scope(list(range(0, num_features)))

    scope_1 = Scope(list(range(0, num_features))[0::2])
    scope_2 = Scope(list(range(0, num_features))[1::2])
    mean = torch.randn(num_features, out_channels, num_repetitions)
    mean_1 = mean[0::2]
    mean_2 = mean[1::2]
    std = torch.rand(num_features, out_channels, num_repetitions)
    std_1 = std[0::2]
    std_2 = std[1::2]
    leaf = make_normal_leaf(scope=scope, mean=mean, std=std)
    leaf_half_1 = make_normal_leaf(scope=scope_1, mean=mean_1, std=std_1)
    leaf_half_2 = make_normal_leaf(scope=scope_2, mean=mean_2, std=std_2)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)
    spn1 = ElementwiseProduct(inputs=split)
    spn2 = ElementwiseProduct(inputs=[leaf_half_1, leaf_half_2])
    assert spn1.out_shape.channels == spn2.out_shape.channels
    assert spn1.out_shape.features == spn2.out_shape.features
    data = make_normal_data(out_features=num_features)
    ll_1 = spn1.log_likelihood(data)
    ll_2 = spn2.log_likelihood(data)
    torch.testing.assert_close(ll_1, ll_2, rtol=1e-5, atol=1e-6)


# New tests for Phase 3 coverage improvement


def test_split_alternate_extra_repr():
    """Test string representation of SplitInterleaved."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    repr_str = split.extra_repr()
    assert isinstance(repr_str, str)
    assert "dim=1" in repr_str


def test_split_alternate_feature_mapping():
    """Test feature_to_scope mapping delegates to input."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    # Split operations delegate to input's feature_to_scope
    feature_scopes = split.feature_to_scope

    # Should be identical to the input's feature_to_scope
    assert feature_scopes.shape == (3, 2, 1)

    assert np.array_equal(
        feature_scopes,
        np.array([[Scope(0), Scope(2), Scope(4)], [Scope(1), Scope(3), Scope(5)]]).reshape(3, 2, 1),
    )

    # Each element should be a Scope object
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_split_alternate_num_splits_one():
    """Test with num_splits=1 (edge case - all features in one group)."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=1, dim=1)

    data = make_normal_data(out_features=6)
    lls = split.log_likelihood(data)

    assert len(lls) == 1
    assert lls[0].shape == (data.shape[0], 6, 3, 1)


def test_split_alternate_num_splits_two():
    """Test optimized path for num_splits=2."""
    scope = Scope(list(range(0, 8)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    data = make_normal_data(out_features=8)
    lls = split.log_likelihood(data)

    assert len(lls) == 2
    # Even indices: 0, 2, 4, 6 -> 4 features
    assert lls[0].shape == (data.shape[0], 4, 2, 1)
    # Odd indices: 1, 3, 5, 7 -> 4 features
    assert lls[1].shape == (data.shape[0], 4, 2, 1)


def test_split_alternate_num_splits_three():
    """Test optimized path for num_splits=3."""
    scope = Scope(list(range(0, 9)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=3, dim=1)

    data = make_normal_data(out_features=9)
    lls = split.log_likelihood(data)

    assert len(lls) == 3
    for ll in lls:
        assert ll.shape == (data.shape[0], 3, 2, 1)


def test_split_alternate_num_splits_four():
    """Test general path for num_splits=4 (uses split_masks)."""
    scope = Scope(list(range(0, 12)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=4, dim=1)

    data = make_normal_data(out_features=12)
    lls = split.log_likelihood(data)

    assert len(lls) == 4
    for ll in lls:
        assert ll.shape == (data.shape[0], 3, 2, 1)


def test_split_alternate_consistency():
    """Test output shapes and values are consistent."""
    scope = Scope(list(range(0, 10)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    data = make_normal_data(out_features=10)
    lls1 = split.log_likelihood(data)
    lls2 = split.log_likelihood(data)

    # Results should be identical for same input
    assert len(lls1) == len(lls2)
    for ll1, ll2 in zip(lls1, lls2):
        torch.testing.assert_close(ll1, ll2, rtol=0.0, atol=0.0)


def test_split_alternate_sampling():
    """Test sampling works correctly with alternating splits."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    n_samples = 20
    data = torch.full((n_samples, 6), torch.nan)
    channel_index = torch.randint(0, 3, size=(n_samples, 6))
    mask = torch.ones((n_samples, 6), dtype=torch.bool)
    rep_index = torch.randint(0, 2, size=(n_samples,))

    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=rep_index)

    samples = split.sample(data=data)

    assert samples.shape == (n_samples, 6)
    assert torch.isfinite(samples).all()


def test_split_alternate_sampling_accepts_split_sized_context():
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    n_samples = 12
    data = torch.full((n_samples, 6), torch.nan)
    # Split-sized context width (features_per_split = 3).
    channel_index = torch.randint(0, 3, size=(n_samples, 3))
    mask = torch.ones((n_samples, 3), dtype=torch.bool)
    sampling_ctx = SamplingContext(
        channel_index=channel_index, mask=mask, repetition_index=torch.zeros(n_samples, dtype=torch.long)
    )

    samples = split.sample(data=data)
    assert samples.shape == (n_samples, 6)
    assert torch.isfinite(samples).all()


def test_split_alternate_rsample_interleaves_split_sized_probability_context():
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    n_samples = 3
    base_probs = torch.tensor(
        [[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]],
        dtype=torch.float32,
    ).repeat(n_samples, 1, 1)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=base_probs,
        mask=torch.ones((n_samples, 3), dtype=torch.bool),
    )

    out = split._rsample(
        data=torch.full((n_samples, 6), torch.nan),
        sampling_ctx=sampling_ctx,
        cache=Cache(),
    )
    if sampling_ctx.sample_accum is not None and sampling_ctx.sample_mass is not None:
        out = sampling_ctx.finalize_with_evidence(out)

    assert torch.isfinite(out).all()
    expected = torch.tensor(
        [[[0.1, 0.9], [0.1, 0.9], [0.2, 0.8], [0.2, 0.8], [0.3, 0.7], [0.3, 0.7]]],
        dtype=torch.float32,
    ).repeat(n_samples, 1, 1)
    torch.testing.assert_close(sampling_ctx.channel_probs, expected, atol=1e-4, rtol=1e-4)


def test_split_alternate_sampling_rejects_incompatible_split_sized_context():
    scope = Scope(list(range(0, 5)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    # input_features=5 is not divisible by num_splits, so split-sized adaptation is invalid.
    data = torch.full((4, 5), torch.nan)
    sampling_ctx = SamplingContext(
        channel_index=torch.randint(0, 2, size=(4, 2)),
        mask=torch.ones((4, 2), dtype=torch.bool),
        repetition_index=torch.zeros(4, dtype=torch.long),
    )
    with pytest.raises(Exception, match="not divisible by num_splits|feature width"):
        split._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())


@pytest.mark.parametrize("num_features", [5, 7, 11])
def test_split_alternate_uneven_features(num_features):
    """Test alternating split with features that don't divide evenly."""
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    data = make_normal_data(out_features=num_features)
    lls = split.log_likelihood(data)

    assert len(lls) == 2
    # Verify split sizes
    expected_size_0 = (num_features + 1) // 2  # Ceiling division
    expected_size_1 = num_features // 2  # Floor division
    assert lls[0].shape[1] == expected_size_0
    assert lls[1].shape[1] == expected_size_1


def test_split_alternate_masks_correctness():
    """Test that split_masks are constructed correctly."""
    scope = Scope(list(range(0, 8)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1)

    # Check masks
    assert len(split.split_masks) == 2
    mask_0 = split.split_masks[0]
    mask_1 = split.split_masks[1]

    # mask_0 should be True for indices 0, 2, 4, 6
    assert mask_0.sum() == 4
    # mask_1 should be True for indices 1, 3, 5, 7
    assert mask_1.sum() == 4

    # Masks should be mutually exclusive
    assert (mask_0 & mask_1).sum() == 0
    # Together they should cover all features
    assert (mask_0 | mask_1).sum() == 8
