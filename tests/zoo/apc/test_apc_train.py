"""Tests for APC training helpers."""

import pytest
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from spflow.exceptions import InvalidParameterError
from spflow.zoo.apc.config import ApcConfig, ApcLossWeights, ApcTrainConfig
from spflow.zoo.apc.decoders import MLPDecoder1D
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
from spflow.zoo.apc.model import AutoencodingPC
from spflow.zoo.apc.train import evaluate_apc, fit_apc, train_apc_step


def _build_model() -> AutoencodingPC:
    encoder = EinetJointEncoder(
        num_x_features=4,
        latent_dim=2,
        num_sums=4,
        num_leaves=4,
        depth=1,
        num_repetitions=1,
        layer_type="linsum",
        structure="top-down",
    )
    decoder = MLPDecoder1D(latent_dim=2, output_dim=4, hidden_dims=(16,))
    config = ApcConfig(
        latent_dim=2,
        rec_loss="mse",
        sample_tau=1.0,
        loss_weights=ApcLossWeights(rec=1.0, kld=0.1, nll=1.0),
    )
    return AutoencodingPC(encoder=encoder, decoder=decoder, config=config)


def test_train_apc_step_updates_parameters_and_returns_metrics():
    torch.manual_seed(30)
    model = _build_model()
    optimizer = Adam(model.parameters(), lr=1e-2)
    batch = torch.randn(12, 4)

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    metrics = train_apc_step(model=model, batch=batch, optimizer=optimizer)

    for key in ("rec", "kld", "nll", "total"):
        assert key in metrics
        assert metrics[key].shape == torch.Size([])
        assert torch.isfinite(metrics[key])
        assert not metrics[key].requires_grad

    changed = any(not torch.equal(before[name], p.detach()) for name, p in model.named_parameters())
    assert changed


def test_evaluate_apc_supports_tensor_and_loader_inputs():
    torch.manual_seed(31)
    model = _build_model()
    data = torch.randn(40, 4)

    metrics_tensor = evaluate_apc(model, data, batch_size=8)
    loader = DataLoader(TensorDataset(data), batch_size=10, shuffle=False)
    metrics_loader = evaluate_apc(model, loader, batch_size=10)

    for metrics in (metrics_tensor, metrics_loader):
        assert set(metrics.keys()) == {"rec", "kld", "nll", "total"}
        for value in metrics.values():
            assert isinstance(value, float)
            assert torch.isfinite(torch.tensor(value))


def test_evaluate_apc_raises_on_empty_input():
    model = _build_model()
    with pytest.raises(InvalidParameterError, match="received no batches"):
        evaluate_apc(model, torch.empty(0, 4), batch_size=8)


def test_fit_apc_smoke_with_validation_data():
    torch.manual_seed(32)
    model = _build_model()
    train = torch.randn(64, 4)
    val = torch.randn(32, 4)

    history = fit_apc(
        model=model,
        train_data=train,
        val_data=val,
        config=ApcTrainConfig(epochs=3, batch_size=16, learning_rate=5e-3),
    )

    assert len(history) == 3
    for epoch_metrics in history:
        expected = {
            "epoch",
            "train_rec",
            "train_kld",
            "train_nll",
            "train_total",
            "val_rec",
            "val_kld",
            "val_nll",
            "val_total",
        }
        assert expected.issubset(epoch_metrics.keys())
        for key in expected:
            assert torch.isfinite(torch.tensor(epoch_metrics[key]))
