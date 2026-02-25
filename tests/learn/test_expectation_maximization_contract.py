"""Cross-implementation EM behavior contracts."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spflow.learn.expectation_maximization import expectation_maximization, expectation_maximization_batched
from tests.contract_data import EM_PARAMS
from tests.utils.leaves import make_data, make_leaf


@pytest.mark.contract
@pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps", EM_PARAMS)
def test_basic_em_contract(leaf_cls, out_features, out_channels, num_reps):
    module = make_leaf(
        cls=leaf_cls,
        out_features=out_features,
        out_channels=out_channels,
        num_repetitions=num_reps,
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=50)

    ll_history = expectation_maximization(module, data, max_steps=3)

    # EM may stop early on convergence, but must never exceed caller-imposed steps.
    assert ll_history.dim() == 1
    assert 1 <= ll_history.shape[0] <= 3
    assert torch.isfinite(ll_history).all()


@pytest.mark.contract
@pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps", EM_PARAMS)
def test_em_changes_parameters_contract(leaf_cls, out_features, out_channels, num_reps):
    module = make_leaf(
        cls=leaf_cls,
        out_features=out_features,
        out_channels=out_channels,
        num_repetitions=num_reps,
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=100)

    params_before = {k: v.clone() for k, v in module.params().items()}
    expectation_maximization(module, data, max_steps=2)

    any_changed = False
    for param_name, param in module.params().items():
        if param.requires_grad and not torch.allclose(param, params_before[param_name], rtol=0.0, atol=0.0):
            any_changed = True
            break

    # Some leaves expose frozen parameters; only enforce updates for trainable ones.
    if any(p.requires_grad for p in module.params().values()):
        assert any_changed, "Expected at least some parameters to change"


@pytest.mark.contract
@pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps", EM_PARAMS)
def test_basic_batched_em_contract(leaf_cls, out_features, out_channels, num_reps):
    module = make_leaf(
        cls=leaf_cls,
        out_features=out_features,
        out_channels=out_channels,
        num_repetitions=num_reps,
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=50)
    dataloader = DataLoader(TensorDataset(data), batch_size=10)

    ll_history = expectation_maximization_batched(module, dataloader, num_epochs=3)

    # Batched API reports one aggregate likelihood per epoch.
    assert ll_history.dim() == 1
    assert ll_history.shape[0] == 3
    assert torch.isfinite(ll_history).all()


@pytest.mark.contract
@pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps", EM_PARAMS)
def test_batched_em_changes_parameters_contract(leaf_cls, out_features, out_channels, num_reps):
    module = make_leaf(
        cls=leaf_cls,
        out_features=out_features,
        out_channels=out_channels,
        num_repetitions=num_reps,
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=100)
    dataloader = DataLoader(TensorDataset(data), batch_size=20)

    params_before = {k: v.clone() for k, v in module.params().items()}
    expectation_maximization_batched(module, dataloader, num_epochs=2)

    any_changed = False
    for param_name, param in module.params().items():
        if param.requires_grad and not torch.allclose(param, params_before[param_name], rtol=0.0, atol=0.0):
            any_changed = True
            break

    # Contract parity with full-batch EM: trainable parameters should move.
    if any(p.requires_grad for p in module.params().values()):
        assert any_changed, "Expected at least some parameters to change"
