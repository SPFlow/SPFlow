"""Tests for APC training helper rollback behavior."""

import pytest
import torch
from torch.optim import Adam

from spflow.exceptions import UnsupportedOperationError
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


def test_train_apc_step_is_unsupported():
    model = _build_model()
    optimizer = Adam(model.parameters(), lr=1e-2)
    batch = torch.randn(12, 4)

    with pytest.raises(UnsupportedOperationError):
        train_apc_step(model=model, batch=batch, optimizer=optimizer)


def test_evaluate_apc_is_unsupported():
    model = _build_model()
    data = torch.randn(40, 4)

    with pytest.raises(UnsupportedOperationError):
        evaluate_apc(model, data, batch_size=8)


def test_fit_apc_is_unsupported():
    model = _build_model()
    train = torch.randn(64, 4)

    with pytest.raises(UnsupportedOperationError):
        fit_apc(
            model=model,
            train_data=train,
            config=ApcTrainConfig(epochs=3, batch_size=16, learning_rate=5e-3),
        )
