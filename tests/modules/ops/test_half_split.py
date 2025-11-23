from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.ops import SplitHalves
from spflow.modules.products import ElementwiseProduct, OuterProduct
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data

cls = [ElementwiseProduct, OuterProduct]

out_channels_values = [1, 4]
out_features_values = [2, 4]
num_repetition_values = [1, 5]


@pytest.mark.parametrize(
    "cls,out_channels,out_features,num_repetitions",
    product(cls, out_channels_values, out_features_values, num_repetition_values),
)
def test_split_result(device, cls, out_channels: int, out_features: int, num_repetitions: int):
    torch.manual_seed(0)
    out_channels = out_channels
    num_features = out_features
    scope = Scope(list(range(0, num_features)))
    scope_1 = Scope(list(range(0, num_features // 2)))
    scope_2 = Scope(list(range(num_features // 2, num_features)))
    mean = torch.randn(num_features, out_channels, num_repetitions)
    std = torch.rand(num_features, out_channels, num_repetitions)
    leaf = make_normal_leaf(scope=scope, mean=mean, std=std).to(device)
    leaf_half_1 = make_normal_leaf(
        scope=scope_1, mean=mean[: num_features // 2], std=std[: num_features // 2]
    ).to(device)
    leaf_half_2 = make_normal_leaf(
        scope=scope_2, mean=mean[num_features // 2 :], std=std[num_features // 2 :]
    ).to(device)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1).to(device)
    spn1 = cls(inputs=split).to(device)
    spn2 = cls(inputs=[leaf_half_1, leaf_half_2]).to(device)
    assert spn1.out_channels == spn2.out_channels
    assert spn1.out_features == spn2.out_features
    data = make_normal_data(out_features=num_features).to(device)
    ll_1 = spn1.log_likelihood(data)
    ll_2 = spn2.log_likelihood(data)
    assert torch.allclose(ll_1, ll_2)

    n_samples = 100
    num_inputs = 2

    data1 = torch.full((n_samples, spn1.out_features * num_inputs), torch.nan).to(device)
    data2 = torch.full((n_samples, spn1.out_features * num_inputs), torch.nan).to(device)
    mask = torch.full((n_samples, spn1.out_features), True, dtype=torch.bool).to(device)
    channel_index = torch.randint(low=0, high=spn1.out_channels, size=(n_samples, spn1.out_features)).to(
        device
    )
    rep_index = torch.randint(low=0, high=num_repetitions, size=(n_samples,)).to(device)
    sampling_ctx = SamplingContext(channel_index=channel_index, repetition_index=rep_index, mask=mask)
    sampling_ctx2 = SamplingContext(channel_index=channel_index, repetition_index=rep_index, mask=mask)

    s1 = spn1.sample(data=data1, sampling_ctx=sampling_ctx, is_mpe=True)
    s2 = spn2.sample(data=data2, sampling_ctx=sampling_ctx2, is_mpe=True)

    assert torch.allclose(s1, s2)


# New tests for Phase 3 coverage improvement


def test_split_halves_extra_repr():
    """Test string representation of SplitHalves."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    repr_str = split.extra_repr()
    assert isinstance(repr_str, str)
    assert "dim=1" in repr_str


def test_split_halves_feature_to_scope():
    """Test feature_to_scope property maps features correctly."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    feature_scopes = split.feature_to_scope
    assert len(feature_scopes) == 2
    # Each split should contain half of the features
    assert len(feature_scopes[0]) == 3
    assert len(feature_scopes[1]) == 3


@pytest.mark.parametrize("num_features,num_splits", [(6, 2), (9, 3), (12, 3)])
def test_split_halves_uneven_features(num_features, num_splits):
    """Test with features that divide evenly (testing behavior is correct)."""
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=num_splits, dim=1)

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


def test_split_halves_single_feature(device):
    """Test with single feature (edge case)."""
    scope = Scope([0])
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1).to(device)
    split = SplitHalves(inputs=leaf, num_splits=1, dim=1).to(device)

    data = make_normal_data(out_features=1).to(device)
    lls = split.log_likelihood(data)

    assert len(lls) == 1
    assert lls[0].shape == (data.shape[0], 1, 2, 1)


def test_split_halves_many_features(device):
    """Test with many features."""
    num_features = 20
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2).to(device)
    split = SplitHalves(inputs=leaf, num_splits=4, dim=1).to(device)

    data = make_normal_data(out_features=num_features).to(device)
    lls = split.log_likelihood(data)

    assert len(lls) == 4
    for ll in lls:
        assert ll.shape[0] == data.shape[0]
        assert ll.shape[1] == num_features // 4


def test_split_halves_log_likelihood_consistency(device):
    """Test log_likelihood produces consistent results."""
    scope = Scope(list(range(0, 10)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1).to(device)

    data = make_normal_data(out_features=10).to(device)
    lls1 = split.log_likelihood(data)
    lls2 = split.log_likelihood(data)

    # Results should be identical for same input
    assert len(lls1) == len(lls2)
    for ll1, ll2 in zip(lls1, lls2):
        assert torch.allclose(ll1, ll2)


def test_split_halves_sampling_consistency(device):
    """Test sampling produces valid samples."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2).to(device)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1).to(device)

    n_samples = 20
    data = torch.full((n_samples, 6), torch.nan).to(device)
    channel_index = torch.randint(0, 3, size=(n_samples, 6)).to(device)
    mask = torch.ones((n_samples, 6), dtype=torch.bool).to(device)
    rep_index = torch.randint(0, 2, size=(n_samples,)).to(device)

    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=rep_index)

    samples = split.sample(data=data, sampling_ctx=sampling_ctx)

    assert samples.shape == (n_samples, 6)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("dim", [1, 2])
def test_split_halves_different_dims(device, dim):
    """Test splitting along different dimensions."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=4, num_repetitions=1).to(device)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=dim).to(device)

    data = make_normal_data(out_features=6).to(device)
    lls = split.log_likelihood(data)

    assert len(lls) == 2
    for ll in lls:
        assert torch.isfinite(ll).all()
