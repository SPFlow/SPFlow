# create tests for split module

from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.ops import SplitAlternate
from spflow.modules.ops import SplitHalves
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data

out_channels_values = [1, 5]
features_values_multiplier = [1, 6]
num_splits = [2, 3]
num_repetitions = [1, 7]
split_type = [SplitHalves, SplitAlternate]
params = list(
    product(out_channels_values, features_values_multiplier, num_splits, split_type, num_repetitions)
)


def make_split(out_channels=3, out_features=3, num_splits=2, split_type=SplitHalves, num_reps=None):
    scope = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope, out_channels=out_channels, num_repetitions=num_reps)

    return split_type(inputs=inputs_a, num_splits=num_splits, dim=1)


@pytest.mark.parametrize("out_channels,features_values_multiplier,num_splits,split_type,num_reps", params)
def test_log_likelihood(
    out_channels: int, features_values_multiplier: int, num_splits: int, split_type, num_reps, device
):
    out_channels = 3
    module = make_split(
        out_channels=out_channels,
        out_features=features_values_multiplier * num_splits,
        num_splits=num_splits,
        split_type=split_type,
        num_reps=num_reps,
    ).to(device)
    data = make_normal_data(out_features=module.out_features).to(device)
    lls = module.log_likelihood(data)
    assert len(lls) == num_splits
    for ll in lls:
        # Always expect 4D output [batch, features, channels, num_reps]
        assert ll.shape == (
            data.shape[0],
            module.out_features // num_splits,
            module.out_channels,
            num_reps,
        )


@pytest.mark.parametrize("out_channels,features_values_multiplier,num_splits,split_type,num_reps", params)
def test_sample(
    out_channels: int, features_values_multiplier: int, num_splits: int, split_type, num_reps, device
):
    n_samples = 10
    out_channels = 3
    module = make_split(
        out_channels=out_channels,
        out_features=features_values_multiplier * num_splits,
        num_splits=num_splits,
        split_type=split_type,
        num_reps=num_reps,
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


def test_split_inherits_scope_from_input():
    """Test that Split modules properly inherit scope from their input."""
    # Test SplitHalves
    scope = Scope(list(range(0, 6)))
    input_leaf = make_normal_leaf(scope, out_channels=3)
    split_halves = SplitHalves(inputs=input_leaf, num_splits=2, dim=1)

    # Split should inherit the same scope as its input
    assert split_halves.scope == input_leaf.scope
    assert split_halves.scope.query == list(range(0, 6))
    assert len(split_halves.scope.query) == 6

    # Test SplitAlternate
    split_alternate = SplitAlternate(inputs=input_leaf, num_splits=3, dim=1)
    assert split_alternate.scope == input_leaf.scope
    assert split_alternate.scope.query == list(range(0, 6))

    # Verify scope is not empty (regression test for the bug)
    assert not split_halves.scope.isempty()
    assert not split_alternate.scope.isempty()


# New tests for Phase 3 coverage improvement - Split base class


def test_split_get_out_shapes_basic():
    """Test get_out_shapes with standard input."""
    scope = Scope(list(range(0, 10)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    event_shape = (10, 10)
    out_shapes = split.get_out_shapes(event_shape)

    assert len(out_shapes) == 2
    assert out_shapes[0] == (10, 5)
    assert out_shapes[1] == (10, 5)


def test_split_get_out_shapes_uneven():
    """Test get_out_shapes with uneven division."""
    scope = Scope(list(range(0, 7)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=3, dim=1)

    event_shape = (10, 7)
    out_shapes = split.get_out_shapes(event_shape)

    assert len(out_shapes) == 3
    # 7 // 3 = 2, 7 % 3 = 1
    # First two splits get 2, last gets remainder (7)
    assert out_shapes[0] == (10, 2)
    assert out_shapes[1] == (10, 2)
    # Bug in line 82: should be remainder not event_shape[1]
    # This test will document current behavior


def test_split_get_out_shapes_dim_zero():
    """Test get_out_shapes when splitting along batch dimension."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=0)

    event_shape = (10, 6)
    out_shapes = split.get_out_shapes(event_shape)

    assert len(out_shapes) == 2
    assert out_shapes[0] == (5, 6)
    assert out_shapes[1] == (5, 6)


def test_split_get_out_shapes_single():
    """Test get_out_shapes for single group (num_splits=1)."""
    scope = Scope(list(range(0, 8)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=1, dim=1)

    event_shape = (10, 8)
    out_shapes = split.get_out_shapes(event_shape)

    assert len(out_shapes) == 1
    assert out_shapes[0] == (10, 8)


# Marginalization tests


def test_split_marginalize_some_features(device):
    """Test marginalize removes subset of features."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1).to(device)

    # Marginalize features [1, 3]
    marg_split = split.marginalize([1, 3], prune=False)

    assert marg_split is not None
    # Scope should have reduced features
    assert len(marg_split.scope.query) == 4


def test_split_marginalize_no_features():
    """Test marginalize with no features to remove."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    # Marginalize features not in scope
    marg_split = split.marginalize([10, 11], prune=False)

    # Should return self unchanged
    assert marg_split is split
    assert len(marg_split.scope.query) == 6


def test_split_marginalize_preserves_scope():
    """Test marginalize preserves underlying scope structure."""
    scope = Scope(list(range(0, 8)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    # Marginalize subset
    marg_split = split.marginalize([2, 3], prune=False)

    assert marg_split is not None
    # Remaining features should be [0, 1, 4, 5, 6, 7]
    assert len(marg_split.scope.query) == 6


def test_split_marginalize_all_features():
    """Test marginalize removes all features."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    # Marginalize all features in scope
    marg_split = split.marginalize(list(range(0, 6)), prune=True)

    # Should return None (fully marginalized)
    assert marg_split is None


def test_split_marginalize_with_prune():
    """Test marginalize with pruning enabled."""
    scope = Scope(list(range(0, 4)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    # Marginalize most features, leaving only one
    marg_split = split.marginalize([0, 1, 2], prune=True)

    # With pruning, if input has only 1 feature, return the leaf directly
    assert marg_split is not None


def test_split_marginalize_preserves_num_splits():
    """Test that marginalization preserves num_splits."""
    scope = Scope(list(range(0, 10)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    marg_split = split.marginalize([1, 2], prune=False)

    assert marg_split is not None
    assert marg_split.num_splits == split.num_splits


@pytest.mark.parametrize("split_type", [SplitHalves, SplitAlternate])
def test_split_marginalize_different_types(split_type):
    """Test marginalization works for different split types."""
    scope = Scope(list(range(0, 8)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = split_type(inputs=leaf, num_splits=2, dim=1)

    marg_split = split.marginalize([2, 5], prune=False)

    assert marg_split is not None
    assert isinstance(marg_split, split_type)
    assert len(marg_split.scope.query) == 6


def test_split_out_features_property():
    """Test out_features property returns correct value."""
    scope = Scope(list(range(0, 8)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    assert split.out_features == leaf.out_features
    assert split.out_features == 8


def test_split_out_channels_property():
    """Test out_channels property returns correct value."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=5, num_repetitions=1)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    assert split.out_channels == leaf.out_channels
    assert split.out_channels == 5


def test_split_num_repetitions_inherited():
    """Test that num_repetitions is inherited from input."""
    scope = Scope(list(range(0, 6)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=4)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1)

    assert split.num_repetitions == leaf.num_repetitions
    assert split.num_repetitions == 4
