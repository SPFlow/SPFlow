"""Expectation maximization implementation-specific and branch tests."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spflow.learn.expectation_maximization import expectation_maximization, expectation_maximization_batched
from spflow.modules import leaves
from tests.utils.leaves import make_data, make_leaf


def test_em_convergence():
    module = make_leaf(cls=leaves.Normal, out_features=2, out_channels=1, num_repetitions=1)
    data = module.distribution().sample((1000,)).squeeze(-1).squeeze(-1)
    ll_history = expectation_maximization(module, data, max_steps=100)
    assert ll_history.shape[0] < 100


@pytest.mark.parametrize("batch_size", [5, 10, 50])
def test_different_batch_sizes(batch_size):
    module = make_leaf(cls=leaves.Normal, out_features=3, out_channels=2, num_repetitions=1)
    data = make_data(cls=leaves.Normal, out_features=3, n_samples=100)
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size)

    ll_history = expectation_maximization_batched(module, dataloader, num_epochs=2)
    assert ll_history.shape[0] == 2
    assert torch.isfinite(ll_history).all()


@pytest.mark.parametrize("num_epochs", [1, 5, 10])
def test_different_num_epochs(num_epochs):
    module = make_leaf(cls=leaves.Normal, out_features=3, out_channels=2, num_repetitions=1)
    data = make_data(cls=leaves.Normal, out_features=3, n_samples=50)
    dataloader = DataLoader(TensorDataset(data), batch_size=10)

    ll_history = expectation_maximization_batched(module, dataloader, num_epochs=num_epochs)
    assert ll_history.shape[0] == num_epochs
    assert torch.isfinite(ll_history).all()


def test_single_batch():
    module = make_leaf(cls=leaves.Normal, out_features=3, out_channels=2, num_repetitions=1)
    data = make_data(cls=leaves.Normal, out_features=3, n_samples=20)
    dataloader = DataLoader(TensorDataset(data), batch_size=100)

    ll_history = expectation_maximization_batched(module, dataloader, num_epochs=3)
    assert ll_history.shape[0] == 3
    assert torch.isfinite(ll_history).all()


def test_empty_dataloader():
    module = make_leaf(cls=leaves.Normal, out_features=3, out_channels=2, num_repetitions=1)
    data = torch.zeros((0, 3))
    dataloader = DataLoader(TensorDataset(data), batch_size=10)

    ll_history = expectation_maximization_batched(module, dataloader, num_epochs=2)
    # Empty epochs intentionally surface NaN aggregates instead of fake finite scores.
    assert ll_history.shape[0] == 2
    assert torch.isnan(ll_history).all()


def test_dataloader_with_raw_tensors():
    module = make_leaf(cls=leaves.Normal, out_features=3, out_channels=2, num_repetitions=1)
    data = make_data(cls=leaves.Normal, out_features=3, n_samples=50)

    class RawTensorDataLoader:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size

        def __iter__(self):
            for i in range(0, len(self.data), self.batch_size):
                yield self.data[i : i + self.batch_size]

    ll_history = expectation_maximization_batched(
        module, RawTensorDataLoader(data, batch_size=10), num_epochs=2
    )
    # Accept non-TensorDataset loaders as long as they yield tensors batch-wise.
    assert ll_history.shape[0] == 2
    assert torch.isfinite(ll_history).all()


def test_dataloader_with_tuple_batches():
    module = make_leaf(cls=leaves.Normal, out_features=3, out_channels=2, num_repetitions=1)
    data = make_data(cls=leaves.Normal, out_features=3, n_samples=50)
    dataloader = DataLoader(TensorDataset(data), batch_size=10)

    ll_history = expectation_maximization_batched(module, dataloader, num_epochs=2)
    assert ll_history.shape[0] == 2
    assert torch.isfinite(ll_history).all()


def test_ll_generally_improves_over_epochs():
    module = make_leaf(cls=leaves.Normal, out_features=2, out_channels=1, num_repetitions=1)
    data = torch.randn(200, 2) * 2 + 1
    dataloader = DataLoader(TensorDataset(data), batch_size=50)

    ll_history = expectation_maximization_batched(module, dataloader, num_epochs=5)
    # Allow small regressions from stochastic batches while guarding against divergence.
    assert ll_history[-1] >= ll_history[0] - 1.0


def test_em_verbose(caplog):
    import logging

    caplog.set_level(logging.INFO)
    module = make_leaf(cls=leaves.Normal, out_features=2, out_channels=1, num_repetitions=1)
    data = make_data(cls=leaves.Normal, out_features=2, n_samples=20)

    expectation_maximization(module, data, max_steps=2, verbose=True)
    assert any("Step" in record.message for record in caplog.records)


def test_batched_em_verbose(caplog):
    import logging

    caplog.set_level(logging.INFO)
    module = make_leaf(cls=leaves.Normal, out_features=2, out_channels=1, num_repetitions=1)
    data = make_data(cls=leaves.Normal, out_features=2, n_samples=20)
    dataloader = DataLoader(TensorDataset(data), batch_size=10)

    expectation_maximization_batched(module, dataloader, num_epochs=2, verbose=True)
    assert any("Epoch" in record.message for record in caplog.records)
