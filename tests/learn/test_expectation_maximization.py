"""Tests for expectation_maximization and expectation_maximization_batched."""

from itertools import product

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spflow.learn.expectation_maximization import (
    expectation_maximization,
    expectation_maximization_batched,
)
from spflow.meta import Scope
from spflow.modules import leaves
from tests.utils.leaves import make_leaf, make_data


# Test configuration parameters
out_features_values = [1, 4]
out_channels_values = [1, 3]
num_repetition_values = [1, 2]
leaf_cls_values = [
    leaves.Normal,
    leaves.Gamma,
    leaves.Poisson,
]
params = list(
    product(
        leaf_cls_values,
        out_features_values,
        out_channels_values,
        num_repetition_values,
    )
)


class TestExpectationMaximization:
    """Tests for the non-batched expectation_maximization function."""

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps", params)
    def test_basic_em(self, leaf_cls, out_features, out_channels, num_reps):
        """Test basic EM runs without errors and returns ll history."""
        module = make_leaf(
            cls=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            num_repetitions=num_reps,
        )
        data = make_data(cls=leaf_cls, out_features=out_features, n_samples=50)

        ll_history = expectation_maximization(module, data, max_steps=3)

        # Check return shape
        assert ll_history.dim() == 1
        assert ll_history.shape[0] >= 1  # At least one step
        assert ll_history.shape[0] <= 3  # At most max_steps

        # Check ll values are finite
        assert torch.isfinite(ll_history).all()

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps", params)
    def test_em_changes_parameters(self, leaf_cls, out_features, out_channels, num_reps):
        """Test that EM updates module parameters."""
        module = make_leaf(
            cls=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            num_repetitions=num_reps,
        )
        data = make_data(cls=leaf_cls, out_features=out_features, n_samples=100)

        # Store parameters before
        params_before = {k: v.clone() for k, v in module.params().items()}

        expectation_maximization(module, data, max_steps=2)

        # Check that at least some parameters changed (if they have gradients)
        any_changed = False
        for param_name, param in module.params().items():
            if param.requires_grad and not torch.allclose(param, params_before[param_name]):
                any_changed = True
                break

        if any(p.requires_grad for p in module.params().values()):
            assert any_changed, "Expected at least some parameters to change"

    def test_em_convergence(self):
        """Test that EM stops early when converged."""
        module = make_leaf(
            cls=leaves.Normal,
            out_features=2,
            out_channels=1,
            num_repetitions=1,
        )
        # Create well-behaved data from the module itself
        data = module.distribution.sample((1000,)).squeeze(-1).squeeze(-1)

        # Run with high max_steps, should converge early
        ll_history = expectation_maximization(module, data, max_steps=100)

        # Should have converged before 100 steps
        assert ll_history.shape[0] < 100


class TestExpectationMaximizationBatched:
    """Tests for the expectation_maximization_batched function."""

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps", params)
    def test_basic_batched_em(self, leaf_cls, out_features, out_channels, num_reps):
        """Test basic batched EM runs and returns ll history per epoch."""
        module = make_leaf(
            cls=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            num_repetitions=num_reps,
        )
        data = make_data(cls=leaf_cls, out_features=out_features, n_samples=50)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        num_epochs = 3
        ll_history = expectation_maximization_batched(
            module, dataloader, num_epochs=num_epochs
        )

        # Check return shape: one entry per epoch
        assert ll_history.dim() == 1
        assert ll_history.shape[0] == num_epochs

        # Check ll values are finite
        assert torch.isfinite(ll_history).all()

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps", params)
    def test_batched_em_changes_parameters(
        self, leaf_cls, out_features, out_channels, num_reps
    ):
        """Test that batched EM updates module parameters."""
        module = make_leaf(
            cls=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            num_repetitions=num_reps,
        )
        data = make_data(cls=leaf_cls, out_features=out_features, n_samples=100)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=20)

        # Store parameters before
        params_before = {k: v.clone() for k, v in module.params().items()}

        expectation_maximization_batched(module, dataloader, num_epochs=2)

        # Check that at least some parameters changed (if they have gradients)
        any_changed = False
        for param_name, param in module.params().items():
            if param.requires_grad and not torch.allclose(param, params_before[param_name]):
                any_changed = True
                break

        if any(p.requires_grad for p in module.params().values()):
            assert any_changed, "Expected at least some parameters to change"

    @pytest.mark.parametrize("batch_size", [5, 10, 50])
    def test_different_batch_sizes(self, batch_size):
        """Test batched EM with different batch sizes."""
        module = make_leaf(
            cls=leaves.Normal,
            out_features=3,
            out_channels=2,
            num_repetitions=1,
        )
        data = make_data(cls=leaves.Normal, out_features=3, n_samples=100)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        ll_history = expectation_maximization_batched(
            module, dataloader, num_epochs=2
        )

        assert ll_history.shape[0] == 2
        assert torch.isfinite(ll_history).all()

    @pytest.mark.parametrize("num_epochs", [1, 5, 10])
    def test_different_num_epochs(self, num_epochs):
        """Test batched EM with different number of epochs."""
        module = make_leaf(
            cls=leaves.Normal,
            out_features=3,
            out_channels=2,
            num_repetitions=1,
        )
        data = make_data(cls=leaves.Normal, out_features=3, n_samples=50)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        ll_history = expectation_maximization_batched(
            module, dataloader, num_epochs=num_epochs
        )

        assert ll_history.shape[0] == num_epochs
        assert torch.isfinite(ll_history).all()

    def test_single_batch(self):
        """Test batched EM when dataloader has only one batch (entire dataset)."""
        module = make_leaf(
            cls=leaves.Normal,
            out_features=3,
            out_channels=2,
            num_repetitions=1,
        )
        data = make_data(cls=leaves.Normal, out_features=3, n_samples=20)
        dataset = TensorDataset(data)
        # Batch size larger than dataset -> single batch
        dataloader = DataLoader(dataset, batch_size=100)

        ll_history = expectation_maximization_batched(
            module, dataloader, num_epochs=3
        )

        assert ll_history.shape[0] == 3
        assert torch.isfinite(ll_history).all()

    def test_empty_dataloader(self):
        """Test batched EM with empty dataloader returns nan."""
        module = make_leaf(
            cls=leaves.Normal,
            out_features=3,
            out_channels=2,
            num_repetitions=1,
        )
        # Empty dataset
        data = torch.zeros((0, 3))
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        ll_history = expectation_maximization_batched(
            module, dataloader, num_epochs=2
        )

        # Should return nan for each epoch since no data
        assert ll_history.shape[0] == 2
        assert torch.isnan(ll_history).all()

    def test_dataloader_with_raw_tensors(self):
        """Test batched EM with dataloader returning raw tensors (not tuples)."""
        module = make_leaf(
            cls=leaves.Normal,
            out_features=3,
            out_channels=2,
            num_repetitions=1,
        )
        data = make_data(cls=leaves.Normal, out_features=3, n_samples=50)

        # Custom dataloader that yields raw tensors instead of tuples
        class RawTensorDataLoader:
            def __init__(self, data, batch_size):
                self.data = data
                self.batch_size = batch_size

            def __iter__(self):
                for i in range(0, len(self.data), self.batch_size):
                    yield self.data[i : i + self.batch_size]

        dataloader = RawTensorDataLoader(data, batch_size=10)

        ll_history = expectation_maximization_batched(
            module, dataloader, num_epochs=2
        )

        assert ll_history.shape[0] == 2
        assert torch.isfinite(ll_history).all()

    def test_dataloader_with_tuple_batches(self):
        """Test batched EM with dataloader returning tuple batches (TensorDataset)."""
        module = make_leaf(
            cls=leaves.Normal,
            out_features=3,
            out_channels=2,
            num_repetitions=1,
        )
        data = make_data(cls=leaves.Normal, out_features=3, n_samples=50)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        ll_history = expectation_maximization_batched(
            module, dataloader, num_epochs=2
        )

        assert ll_history.shape[0] == 2
        assert torch.isfinite(ll_history).all()

    def test_ll_generally_improves_over_epochs(self):
        """Test that log-likelihood generally improves or stays similar over epochs."""
        module = make_leaf(
            cls=leaves.Normal,
            out_features=2,
            out_channels=1,
            num_repetitions=1,
        )
        # Create data from a different distribution to ensure learning happens
        torch.manual_seed(42)
        data = torch.randn(200, 2) * 2 + 1  # Different mean and std
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=50)

        ll_history = expectation_maximization_batched(
            module, dataloader, num_epochs=5
        )

        # Later epochs should have similar or better (higher) LL than first epoch
        # Allow some tolerance since mini-batch updates can be noisy
        assert ll_history[-1] >= ll_history[0] - 1.0


class TestVerboseMode:
    """Tests for verbose logging in EM functions."""

    def test_em_verbose(self, caplog):
        """Test that verbose mode logs progress for non-batched EM."""
        import logging

        caplog.set_level(logging.INFO)

        module = make_leaf(
            cls=leaves.Normal,
            out_features=2,
            out_channels=1,
            num_repetitions=1,
        )
        data = make_data(cls=leaves.Normal, out_features=2, n_samples=20)

        expectation_maximization(module, data, max_steps=2, verbose=True)

        # Check that some logging occurred
        assert any("Step" in record.message for record in caplog.records)

    def test_batched_em_verbose(self, caplog):
        """Test that verbose mode logs progress for batched EM."""
        import logging

        caplog.set_level(logging.INFO)

        module = make_leaf(
            cls=leaves.Normal,
            out_features=2,
            out_channels=1,
            num_repetitions=1,
        )
        data = make_data(cls=leaves.Normal, out_features=2, n_samples=20)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        expectation_maximization_batched(
            module, dataloader, num_epochs=2, verbose=True
        )

        # Check that some logging occurred
        assert any("Epoch" in record.message for record in caplog.records)
