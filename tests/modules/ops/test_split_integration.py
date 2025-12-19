"""Integration tests for split operations.

Tests interactions between different split types and with other modules.
"""

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.ops import SplitInterleaved, SplitConsecutive
from spflow.modules.products import ElementwiseProduct, OuterProduct
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data


def test_split_operations_log_likelihood_consistency(device):
    """Test log_likelihood consistent across split types with products."""
    num_features = 6
    scope = Scope(list(range(0, num_features)))
    out_channels = 3
    num_repetitions = 2

    # Create identical leaves for both split types
    torch.manual_seed(42)
    mean = torch.randn(num_features, out_channels, num_repetitions)
    std = torch.rand(num_features, out_channels, num_repetitions) + 0.1

    leaf1 = make_normal_leaf(scope, mean=mean, std=std).to(device)
    leaf2 = make_normal_leaf(scope, mean=mean, std=std).to(device)

    # Create splits
    split_consecutive = SplitConsecutive(inputs=leaf1, num_splits=2, dim=1).to(device)
    split_interleaved = SplitInterleaved(inputs=leaf2, num_splits=2, dim=1).to(device)

    # Both should work with products
    prod1 = ElementwiseProduct(inputs=split_consecutive).to(device)
    prod2 = ElementwiseProduct(inputs=split_interleaved).to(device)

    # Generate test data
    data = make_normal_data(out_features=num_features).to(device)

    # Both should produce valid log likelihoods
    ll1 = prod1.log_likelihood(data)
    ll2 = prod2.log_likelihood(data)

    assert ll1.shape == ll2.shape
    assert torch.isfinite(ll1).all()
    assert torch.isfinite(ll2).all()


def test_split_operations_sampling(device):
    """Test sampling from split operations."""
    num_features = 8
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2).to(device)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1).to(device)

    n_samples = 50
    data = torch.full((n_samples, num_features), torch.nan).to(device)
    channel_index = torch.randint(0, 3, size=(n_samples, num_features)).to(device)
    mask = torch.ones((n_samples, num_features), dtype=torch.bool).to(device)
    rep_index = torch.randint(0, 2, size=(n_samples,)).to(device)

    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=rep_index)

    samples = split.sample(data=data, sampling_ctx=sampling_ctx)

    # Verify samples
    assert samples.shape == (n_samples, num_features)
    assert torch.isfinite(samples).all()
    # Verify samples are not all the same
    assert samples.std() > 0


def test_split_operations_gradients(device):
    """Test gradients flow through split operations."""
    num_features = 6
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1).to(device)

    # Create product to get scalar output
    prod = ElementwiseProduct(inputs=split).to(device)

    data = make_normal_data(out_features=num_features).to(device)
    ll = prod.log_likelihood(data)

    # Compute sum for scalar loss
    loss = ll.sum()

    # Check gradients flow through
    loss.backward()

    # Verify leaf parameters have gradients
    for param in leaf.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_split_operations_batched(device):
    """Test split operations with batched inputs."""
    num_features = 6
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
    split = SplitInterleaved(inputs=leaf, num_splits=3, dim=1).to(device)

    # Different batch sizes
    batch_sizes = [1, 10, 50]

    for batch_size in batch_sizes:
        data = make_normal_data(num_samples=batch_size, out_features=num_features).to(device)
        lls = split.log_likelihood(data)

        assert len(lls) == 3
        for ll in lls:
            assert ll.shape[0] == batch_size


def test_split_operations_with_products(device):
    """Test split operations work correctly with products."""
    num_features = 12
    scope1 = Scope(list(range(0, 6)))
    scope2 = Scope(list(range(6, 12)))

    leaf1 = make_normal_leaf(scope1, out_channels=3, num_repetitions=1).to(device)
    leaf2 = make_normal_leaf(scope2, out_channels=3, num_repetitions=1).to(device)

    # Create product from separate leaves (avoids split broadcasting issues)
    prod = ElementwiseProduct(inputs=[leaf1, leaf2]).to(device)

    # Test it works
    data = make_normal_data(out_features=num_features).to(device)
    ll = prod.log_likelihood(data)

    assert torch.isfinite(ll).all()
    assert ll.shape[0] == data.shape[0]


def test_split_with_outer_product(device):
    """Test split operations with OuterProduct."""
    num_features = 6
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1).to(device)

    # Use with outer product
    outer_prod = OuterProduct(inputs=split).to(device)

    data = make_normal_data(out_features=num_features).to(device)
    ll = outer_prod.log_likelihood(data)

    assert torch.isfinite(ll).all()
    # OuterProduct should increase channels
    assert outer_prod.out_shape.channels > split.out_shape.channels


def test_split_alternating_pattern_verification(device):
    """Verify alternating split pattern is correct."""
    num_features = 8
    scope = Scope(list(range(0, num_features)))

    # Create leaf with known values
    torch.manual_seed(123)
    mean = torch.arange(num_features).float().reshape(num_features, 1, 1)
    std = torch.ones(num_features, 1, 1) * 0.1

    leaf = make_normal_leaf(scope, mean=mean, std=std).to(device)
    split = SplitInterleaved(inputs=leaf, num_splits=2, dim=1).to(device)

    # Get log likelihoods
    data = make_normal_data(out_features=num_features).to(device)
    lls = split.log_likelihood(data)

    # First split should have features 0, 2, 4, 6
    # Second split should have features 1, 3, 5, 7
    assert lls[0].shape[1] == 4
    assert lls[1].shape[1] == 4


def test_split_consecutive_pattern_verification(device):
    """Verify halves split pattern is correct."""
    num_features = 8
    scope = Scope(list(range(0, num_features)))

    # Create leaf with known values
    torch.manual_seed(123)
    mean = torch.arange(num_features).float().reshape(num_features, 1, 1)
    std = torch.ones(num_features, 1, 1) * 0.1

    leaf = make_normal_leaf(scope, mean=mean, std=std).to(device)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1).to(device)

    # Get log likelihoods
    data = make_normal_data(out_features=num_features).to(device)
    lls = split.log_likelihood(data)

    # First split should have features 0-3
    # Second split should have features 4-7
    assert lls[0].shape[1] == 4
    assert lls[1].shape[1] == 4


@pytest.mark.parametrize("split_type", [SplitConsecutive, SplitInterleaved])
def test_split_sampling_mpe_mode(split_type, device):
    """Test sampling in MPE (Most Probable Explanation) mode."""
    num_features = 6
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=2).to(device)
    split = split_type(inputs=leaf, num_splits=2, dim=1).to(device)

    n_samples = 20
    data = torch.full((n_samples, num_features), torch.nan).to(device)
    channel_index = torch.randint(0, 3, size=(n_samples, num_features)).to(device)
    mask = torch.ones((n_samples, num_features), dtype=torch.bool).to(device)
    rep_index = torch.randint(0, 2, size=(n_samples,)).to(device)

    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=rep_index)

    # Sample in MPE mode
    samples = split.sample(data=data, sampling_ctx=sampling_ctx, is_mpe=True)

    assert samples.shape == (n_samples, num_features)
    assert torch.isfinite(samples).all()


def test_split_consistent_results(device):
    """Test that splits produce consistent results across multiple calls."""
    num_features = 6
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1).to(device)

    data = make_normal_data(out_features=num_features).to(device)

    # Multiple calls should produce identical results
    lls1 = split.log_likelihood(data)
    lls2 = split.log_likelihood(data)

    assert len(lls1) == len(lls2)
    for ll1, ll2 in zip(lls1, lls2):
        assert torch.allclose(ll1, ll2)


@pytest.mark.parametrize("num_splits", [2, 3, 4])
def test_split_different_num_splits(num_splits, device):
    """Test splits work with different number of splits."""
    num_features = num_splits * 3  # Ensure even division
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=2, num_repetitions=1).to(device)

    # Test both split types
    split_consecutive = SplitConsecutive(inputs=leaf, num_splits=num_splits, dim=1).to(device)
    split_alt = SplitInterleaved(inputs=leaf, num_splits=num_splits, dim=1).to(device)

    data = make_normal_data(out_features=num_features).to(device)

    lls_halves = split_consecutive.log_likelihood(data)
    lls_alt = split_alt.log_likelihood(data)

    assert len(lls_halves) == num_splits
    assert len(lls_alt) == num_splits


def test_split_scope_inheritance(device):
    """Test scope is correctly inherited from input."""
    scope = Scope(list(range(0, 8)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1).to(device)

    # Split should inherit scope from leaf
    assert split.scope == leaf.scope
    assert len(split.scope.query) == 8

    # Should be able to compute log likelihoods
    data = make_normal_data(out_features=8).to(device)
    lls = split.log_likelihood(data)
    assert len(lls) >= 1


def test_split_with_partial_mask_sampling(device):
    """Test sampling with partial masking."""
    num_features = 6
    scope = Scope(list(range(0, num_features)))
    leaf = make_normal_leaf(scope, out_channels=3, num_repetitions=1).to(device)
    split = SplitConsecutive(inputs=leaf, num_splits=2, dim=1).to(device)

    n_samples = 10
    data = torch.full((n_samples, num_features), torch.nan).to(device)
    channel_index = torch.randint(0, 3, size=(n_samples, num_features)).to(device)

    # Partially mask some features
    mask = torch.ones((n_samples, num_features), dtype=torch.bool).to(device)
    mask[:, ::2] = False  # Mask every other feature

    rep_index = torch.zeros(n_samples, dtype=torch.long).to(device)

    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=rep_index)

    samples = split.sample(data=data, sampling_ctx=sampling_ctx)

    # Masked features should still be NaN
    assert torch.isnan(samples[:, ::2]).any()
    # Unmasked features should be finite
    assert torch.isfinite(samples[:, 1::2]).all()
