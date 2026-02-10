"""Targeted rollback coverage for APC modules."""

from __future__ import annotations

import pytest
import torch

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.decoders import MLPDecoder1D
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
from spflow.zoo.apc.model import AutoencodingPC


def _build_einet() -> EinetJointEncoder:
    return EinetJointEncoder(
        num_x_features=4,
        latent_dim=2,
        num_sums=4,
        num_leaves=4,
        depth=1,
        num_repetitions=1,
        layer_type="linsum",
        structure="top-down",
    )


def _build_convpc() -> ConvPcJointEncoder:
    return ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=4,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
    )


def test_einet_encode_decode_paths_still_work() -> None:
    enc = _build_einet()
    x = torch.randn(3, 4)
    z = enc.encode(x, tau=0.7)
    x_rec = enc.decode(z, tau=0.7)

    assert z.shape == (3, 2)
    assert x_rec.shape == (3, 4)
    assert torch.isfinite(z).all()
    assert torch.isfinite(x_rec).all()


def test_convpc_encode_decode_paths_still_work() -> None:
    enc = _build_convpc()
    x = torch.randn(3, 1, 4, 4)
    z = enc.encode(x, tau=0.7)
    x_rec = enc.decode(z, tau=0.7)

    assert z.shape == (3, 4)
    assert x_rec.shape == (3, 1, 4, 4)
    assert torch.isfinite(z).all()
    assert torch.isfinite(x_rec).all()


def test_encoder_latent_stats_apis_are_unsupported() -> None:
    einet = _build_einet()
    convpc = _build_convpc()
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 1, 4, 4)

    with pytest.raises(UnsupportedOperationError):
        einet.encode(x1, return_latent_stats=True)
    with pytest.raises(UnsupportedOperationError):
        einet.latent_stats(x1)

    with pytest.raises(UnsupportedOperationError):
        convpc.encode(x2, return_latent_stats=True)
    with pytest.raises(UnsupportedOperationError):
        convpc.latent_stats(x2)


def test_apc_loss_and_training_helpers_are_unsupported() -> None:
    cfg = ApcConfig(latent_dim=2, rec_loss="mse", sample_tau=1.0, loss_weights=ApcLossWeights())
    model = AutoencodingPC(
        encoder=_build_einet(),
        decoder=MLPDecoder1D(latent_dim=2, output_dim=4, hidden_dims=(8,)),
        config=cfg,
    )

    with pytest.raises(UnsupportedOperationError):
        model.loss_components(torch.randn(2, 4))
    with pytest.raises(UnsupportedOperationError):
        model.loss(torch.randn(2, 4))


def test_convpc_sample_prior_validation_still_active() -> None:
    enc = _build_convpc()
    with pytest.raises(InvalidParameterError, match="num_samples must be >= 1"):
        enc.sample_prior_z(0)
