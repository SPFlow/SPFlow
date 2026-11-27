"""Expectation maximization branch and regression tests."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spflow.exceptions import NumericalInstabilityError
from spflow.learn.expectation_maximization import expectation_maximization, expectation_maximization_batched
from spflow.meta import Scope
from spflow.modules import leaves
from spflow.zoo.rat import RatSPN
from tests.test_helpers.builders import make_einet
from tests.utils.leaves import make_data, make_leaf


def _make_em_regression_model():
    torch.manual_seed(7)
    return make_einet(
        num_features=8,
        num_classes=1,
        num_sums=4,
        num_leaves=3,
        depth=3,
        num_repetitions=2,
        layer_type="einsum",
        structure="top-down",
    )


def _make_em_regression_data(module: torch.nn.Module, num_samples: int = 64) -> torch.Tensor:
    torch.manual_seed(11)
    with torch.no_grad():
        return module.sample(num_samples=num_samples)


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


def test_full_batch_em_regression_on_representative_einet():
    module = _make_em_regression_model()
    data = _make_em_regression_data(module)

    ll_history = expectation_maximization(module, data, max_steps=2)

    assert ll_history.shape == (2,)
    assert torch.isfinite(ll_history).all()


def test_batched_em_regression_on_representative_einet():
    module = _make_em_regression_model()
    data = _make_em_regression_data(module, num_samples=96)
    dataloader = DataLoader(TensorDataset(data), batch_size=24, shuffle=False)

    ll_history = expectation_maximization_batched(module, dataloader, num_epochs=1)

    assert ll_history.shape == (1,)
    assert torch.isfinite(ll_history).all()


def _make_poisoned_bernoulli_rat(kind: str) -> RatSPN:
    torch.manual_seed(0)
    leaf = leaves.Bernoulli(scope=Scope(list(range(4))), out_channels=2, num_repetitions=2)
    model = RatSPN(leaf_modules=[leaf], n_root_nodes=1, n_region_nodes=2, num_repetitions=2, depth=1)
    with torch.no_grad():
        if kind == "nan_sum_weights":
            model.root_node.logits.fill_(float("nan"))
        else:  # "bernoulli_p_one": p == 1 exactly, so every x = 0 has zero likelihood
            leaf.logits.fill_(float("inf"))
    return model


_BINARY_DATA = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]).repeat(16, 1)


@pytest.mark.parametrize("kind", ["nan_sum_weights", "bernoulli_p_one"])
def test_em_raises_on_non_finite_log_likelihood(kind):
    """A NaN log-likelihood never satisfies the convergence check, so EM used to iterate until max_steps."""
    model = _make_poisoned_bernoulli_rat(kind)
    with pytest.raises(NumericalInstabilityError, match="EM step 0"):
        expectation_maximization(model, _BINARY_DATA, max_steps=20)


@pytest.mark.parametrize("kind", ["nan_sum_weights", "bernoulli_p_one"])
def test_batched_em_raises_on_non_finite_log_likelihood(kind):
    """Batched EM used to finish its epochs silently and return a NaN history."""
    model = _make_poisoned_bernoulli_rat(kind)
    loader = DataLoader(TensorDataset(_BINARY_DATA), batch_size=16)
    with pytest.raises(NumericalInstabilityError, match="EM epoch 0, batch 0"):
        expectation_maximization_batched(model, loader, num_epochs=2)
