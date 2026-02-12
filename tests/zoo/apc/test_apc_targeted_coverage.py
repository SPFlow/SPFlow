"""Targeted rollback coverage for APC modules."""

from __future__ import annotations

import pytest
import torch

from spflow.exceptions import InvalidParameterError, ShapeError, UnsupportedOperationError
from spflow.meta.data import Scope
from spflow.modules.leaves import Normal
from spflow.utils.sampling_context import SamplingContext
from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.decoders import MLPDecoder1D
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder, _LatentSelectionCapture
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


def test_latent_selection_capture_raises_on_unsupported_context_widths() -> None:
    leaf = Normal(scope=Scope([0, 1, 2, 3]), out_channels=1)
    capture = _LatentSelectionCapture(inputs=leaf, capture_fn=lambda *args: None)

    ctx = SamplingContext(
        channel_index=torch.zeros((2, 3), dtype=torch.long),
        mask=torch.ones((2, 3), dtype=torch.bool),
    )
    with pytest.raises(ShapeError, match="channel width mismatch"):
        capture._capture(ctx, data_num_features=6)

    ctx2 = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
        repetition_index=torch.zeros((2, 3), dtype=torch.long),
    )
    with pytest.raises(ShapeError, match="repetition width mismatch"):
        capture._capture(ctx2, data_num_features=6)


def test_convpc_latent_repetition_indices_reject_unsupported_widths() -> None:
    enc = _build_convpc()
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
        repetition_index=torch.zeros((2, 2), dtype=torch.long),
    )
    with pytest.raises(ShapeError, match="Latent repetition index feature mismatch"):
        enc._resolve_latent_repetition_indices(
            sampling_ctx=ctx,
            batch_size=2,
            loc=torch.zeros((2, 1, 3)),
        )
