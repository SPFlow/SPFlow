"""Tests for SplitByIndex module."""

from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.ops import SplitByIndex, SplitMode
from spflow.modules.products import ElementwiseProduct, OuterProduct
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data


class TestSplitByIndexBasic:
    """Basic tests for SplitByIndex functionality."""

    def test_create_split_by_index(self, device):
        """Test basic SplitByIndex creation."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        split = SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4, 5]]).to(device)

        assert split.num_splits == 2
        assert split.indices == [[0, 1, 2], [3, 4, 5]]
        assert split.scope == leaf.scope

    def test_uneven_splits(self, device):
        """Test SplitByIndex with uneven split sizes."""
        scope = Scope(list(range(0, 8)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        # Split into groups of different sizes
        split = SplitByIndex(inputs=leaf, indices=[[0, 1], [2, 3, 4], [5, 6, 7]]).to(device)

        assert split.num_splits == 3
        assert split.indices == [[0, 1], [2, 3, 4], [5, 6, 7]]

    def test_non_contiguous_indices(self, device):
        """Test SplitByIndex with non-contiguous indices."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        # Non-contiguous splits
        split = SplitByIndex(inputs=leaf, indices=[[0, 2, 4], [1, 3, 5]]).to(device)

        assert split.num_splits == 2
        assert split.indices == [[0, 2, 4], [1, 3, 5]]


class TestSplitByIndexValidation:
    """Tests for SplitByIndex validation."""

    def test_overlapping_indices_raises(self, device):
        """Test that overlapping indices raise ValueError."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        with pytest.raises(ValueError, match="duplicates"):
            SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [2, 3, 4, 5]])

    def test_missing_indices_raises(self, device):
        """Test that missing indices raise ValueError."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        with pytest.raises(ValueError, match="Missing"):
            SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4]])  # Missing 5

    def test_out_of_bounds_raises(self, device):
        """Test that out of bounds indices raise ValueError."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        with pytest.raises(ValueError, match="out of bounds"):
            SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4, 10]])

    def test_negative_indices_raises(self, device):
        """Test that negative indices raise ValueError."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        with pytest.raises(ValueError, match="out of bounds"):
            SplitByIndex(inputs=leaf, indices=[[0, 1, -1], [3, 4, 5]])

    def test_none_indices_raises(self, device):
        """Test that None indices raise ValueError."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        with pytest.raises(ValueError, match="indices must be provided"):
            SplitByIndex(inputs=leaf, indices=None)


class TestSplitByIndexLogLikelihood:
    """Tests for SplitByIndex log_likelihood computation."""

    def test_log_likelihood_shape(self, device):
        """Test log_likelihood returns correct shapes."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4, 5]]).to(device)

        data = make_normal_data(out_features=6).to(device)
        lls = split.log_likelihood(data)

        assert len(lls) == 2
        assert lls[0].shape == (data.shape[0], 3, 3, 2)
        assert lls[1].shape == (data.shape[0], 3, 3, 2)

    def test_log_likelihood_uneven_splits(self, device):
        """Test log_likelihood with uneven split sizes."""
        scope = Scope(list(range(0, 8)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1], [2, 3, 4, 5, 6, 7]]).to(device)

        data = make_normal_data(out_features=8).to(device)
        lls = split.log_likelihood(data)

        assert len(lls) == 2
        assert lls[0].shape == (data.shape[0], 2, 3, 1)
        assert lls[1].shape == (data.shape[0], 6, 3, 1)

    def test_log_likelihood_correct_indices(self, device):
        """Test that log_likelihood selects correct features."""
        scope = Scope(list(range(0, 4)))

        # Create leaf with known distinct values per feature
        torch.manual_seed(42)
        mean = torch.arange(4).float().reshape(4, 1, 1)
        std = torch.ones(4, 1, 1) * 10.0
        leaf = make_normal_leaf(scope, mean=mean, std=std).to(device)

        # Split with non-contiguous indices
        split = SplitByIndex(inputs=leaf, indices=[[0, 2], [1, 3]]).to(device)

        # Data with distinct values
        data = mean.squeeze().reshape(1, 4).to(device)
        lls = split.log_likelihood(data)

        # First split should get features 0, 2
        # Second split should get features 1, 3
        # At the mean, log_likelihood should be highest
        assert len(lls) == 2
        assert lls[0].shape[1] == 2  # features 0, 2
        assert lls[1].shape[1] == 2  # features 1, 3


class TestSplitByIndexSampling:
    """Tests for SplitByIndex sampling."""

    @pytest.mark.parametrize("num_reps", [1, 3])
    def test_sample_basic(self, num_reps, device):
        """Test basic sampling functionality."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=num_reps).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4, 5]]).to(device)

        n_samples = 20
        data = torch.full((n_samples, 6), torch.nan).to(device)
        channel_index = torch.randint(0, 3, size=(n_samples, 6)).to(device)
        mask = torch.ones((n_samples, 6), dtype=torch.bool).to(device)
        rep_index = torch.randint(0, num_reps, size=(n_samples,)).to(device)

        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=rep_index)

        samples = split.sample(data=data, sampling_ctx=sampling_ctx)

        assert samples.shape == (n_samples, 6)
        assert torch.isfinite(samples).all()

    def test_sample_non_contiguous(self, device):
        """Test sampling with non-contiguous indices."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 2, 4], [1, 3, 5]]).to(device)

        n_samples = 20
        data = torch.full((n_samples, 6), torch.nan).to(device)
        channel_index = torch.randint(0, 3, size=(n_samples, 6)).to(device)
        mask = torch.ones((n_samples, 6), dtype=torch.bool).to(device)
        rep_index = torch.zeros(n_samples, dtype=torch.long).to(device)

        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=rep_index)

        samples = split.sample(data=data, sampling_ctx=sampling_ctx)

        assert samples.shape == (n_samples, 6)
        assert torch.isfinite(samples).all()


class TestSplitByIndexMerge:
    """Tests for SplitByIndex merge_split_indices method."""

    def test_merge_contiguous(self, device):
        """Test merge_split_indices with contiguous splits."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4, 5]]).to(device)

        # Create split indices
        batch_size = 5
        left_indices = torch.arange(3).unsqueeze(0).expand(batch_size, -1).to(device)
        right_indices = torch.arange(3, 6).unsqueeze(0).expand(batch_size, -1).to(device)

        merged = split.merge_split_indices(left_indices, right_indices)

        assert merged.shape == (batch_size, 6)
        # Should be [0,1,2,3,4,5] for each batch
        expected = torch.arange(6).unsqueeze(0).expand(batch_size, -1).to(device)
        assert torch.equal(merged, expected)

    def test_merge_non_contiguous(self, device):
        """Test merge_split_indices with non-contiguous splits."""
        scope = Scope(list(range(0, 4)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 2], [1, 3]]).to(device)

        batch_size = 3
        # First split (indices 0, 2) gets values 10, 20
        # Second split (indices 1, 3) gets values 30, 40
        first_split = torch.tensor([[10, 20], [10, 20], [10, 20]]).to(device)
        second_split = torch.tensor([[30, 40], [30, 40], [30, 40]]).to(device)

        merged = split.merge_split_indices(first_split, second_split)

        # Result should be [10, 30, 20, 40] for each batch (original order: 0, 1, 2, 3)
        expected = torch.tensor([[10, 30, 20, 40], [10, 30, 20, 40], [10, 30, 20, 40]]).to(device)
        assert torch.equal(merged, expected)


class TestSplitModeByIndex:
    """Tests for SplitMode.by_index factory method."""

    def test_split_mode_by_index_creation(self, device):
        """Test SplitMode.by_index creates correct configuration."""
        mode = SplitMode.by_index(indices=[[0, 1], [2, 3, 4], [5]])

        assert mode.split_type == "by_index"
        assert mode.num_splits == 3
        assert mode.indices == [[0, 1], [2, 3, 4], [5]]

    def test_split_mode_by_index_create(self, device):
        """Test SplitMode.by_index.create() produces SplitByIndex module."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)

        mode = SplitMode.by_index(indices=[[0, 1, 2], [3, 4, 5]])
        split = mode.create(leaf)

        assert isinstance(split, SplitByIndex)
        assert split.indices == [[0, 1, 2], [3, 4, 5]]

    def test_split_mode_by_index_repr(self):
        """Test SplitMode.by_index repr."""
        mode = SplitMode.by_index(indices=[[0, 1], [2, 3]])
        assert "by_index" in repr(mode)
        assert "[[0, 1], [2, 3]]" in repr(mode)


class TestSplitByIndexIntegration:
    """Integration tests for SplitByIndex with other modules."""

    def test_with_elementwise_product(self, device):
        """Test SplitByIndex works with ElementwiseProduct."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4, 5]]).to(device)
        prod = ElementwiseProduct(inputs=split).to(device)

        data = make_normal_data(out_features=6).to(device)
        ll = prod.log_likelihood(data)

        assert torch.isfinite(ll).all()
        assert ll.shape[0] == data.shape[0]

    def test_with_outer_product(self, device):
        """Test SplitByIndex works with OuterProduct."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4, 5]]).to(device)
        prod = OuterProduct(inputs=split).to(device)

        data = make_normal_data(out_features=6).to(device)
        ll = prod.log_likelihood(data)

        assert torch.isfinite(ll).all()
        # OuterProduct should increase channels
        assert prod.out_shape.channels > split.out_shape.channels

    def test_gradients_flow(self, device):
        """Test gradients flow through SplitByIndex."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1, 2], [3, 4, 5]]).to(device)
        prod = ElementwiseProduct(inputs=split).to(device)

        data = make_normal_data(out_features=6).to(device)
        ll = prod.log_likelihood(data)
        loss = ll.sum()
        loss.backward()

        for param in leaf.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSplitByIndexProperties:
    """Tests for SplitByIndex property methods."""

    def test_scope_inherited(self, device):
        """Test that scope is correctly inherited from input."""
        scope = Scope(list(range(0, 8)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1, 2, 3], [4, 5, 6, 7]]).to(device)

        assert split.scope == leaf.scope
        assert len(split.scope.query) == 8

    def test_out_shape_inherited(self, device):
        """Test that out_shape is correctly computed."""
        scope = Scope(list(range(0, 6)))
        leaf = make_normal_leaf(scope, out_channels=5, num_repetitions=3).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1], [2, 3], [4, 5]]).to(device)

        # out_shape should match input
        assert split.out_shape.features == 6
        assert split.out_shape.channels == 5
        assert split.out_shape.repetitions == 3

    def test_extra_repr(self, device):
        """Test extra_repr includes indices."""
        scope = Scope(list(range(0, 4)))
        leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
        split = SplitByIndex(inputs=leaf, indices=[[0, 1], [2, 3]]).to(device)

        repr_str = split.extra_repr()
        assert "indices" in repr_str
        assert "[[0, 1], [2, 3]]" in repr_str
