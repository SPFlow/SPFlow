"""Architecture parity checks for APC Conv-PC reference mode."""

from __future__ import annotations

import pytest
import torch

from spflow.modules.conv.prod_conv import ProdConv
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.modules.sums import Sum
from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.apc.decoders import NeuralDecoder2D
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder


def _build_reference_encoder(*, latent_dim: int, perm_latents: bool = False) -> ConvPcJointEncoder:
    return ConvPcJointEncoder(
        input_height=32,
        input_width=32,
        input_channels=1,
        latent_dim=latent_dim,
        channels=64,
        depth=4,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
        perm_latents=perm_latents,
    )


def test_convpc_reference_default_topology_matches_expected_stage_counts() -> None:
    encoder = _build_reference_encoder(latent_dim=64)

    assert encoder.architecture == "reference"
    assert encoder.latent_sum_layer is not None
    assert (
        encoder._latent_target_features == 64
    )  # Guards the reference-depth contract for latent injection width.
    assert len(encoder.latent_prod_layers) == 0

    prod_layers = [m for m in encoder.layers if isinstance(m, ProdConv)]
    sum_layers = [m for m in encoder.layers if isinstance(m, Sum)]
    fusion_layers = [m for m in encoder.layers if isinstance(m, ElementwiseProduct)]

    # Keep stage counts fixed so architecture refactors cannot silently drift.
    # Fusion should replace exactly one product-stage output in reference mode.
    assert len(prod_layers) == 3
    assert len(sum_layers) == 4
    assert len(fusion_layers) == 1

    x = torch.randn(3, 1, 32, 32)
    z = encoder.encode(x, tau=1.0)
    assert z.shape == (3, 64)
    assert torch.isfinite(z).all()


def test_convpc_reference_latent_dim_larger_than_injection_uses_latent_prod_layers() -> None:
    encoder = _build_reference_encoder(latent_dim=256)
    assert (
        len(encoder.latent_prod_layers) == 2
    )  # Verifies down-projection depth when latent_dim exceeds injection width.

    x = torch.randn(4, 1, 32, 32)
    stats, z = encoder.encode(x, return_latent_stats=True)
    assert isinstance(stats, LatentStats)
    assert stats.mu.shape == (4, 256)
    assert stats.logvar.shape == (4, 256)
    assert stats.kld_per_sample.shape == (4,)
    assert stats.decode_latent.shape == (4, 256)
    assert z.shape == (4, 256)
    assert torch.isfinite(stats.mu).all()
    assert torch.isfinite(stats.logvar).all()
    assert torch.isfinite(stats.kld_per_sample).all()
    assert torch.isfinite(stats.decode_latent).all()
    assert torch.isfinite(z).all()


def test_convpc_reference_latent_dim_smaller_than_injection_uses_packing_and_perm_buffers() -> None:
    no_perm = _build_reference_encoder(latent_dim=16, perm_latents=False)
    assert len(no_perm.latent_prod_layers) == 0
    assert no_perm.latent_perm is None
    assert no_perm.latent_perm_inv is None

    x = torch.randn(2, 1, 32, 32)
    z = no_perm.encode(x, tau=1.0)
    assert z.shape == (2, 16)
    assert torch.isfinite(z).all()

    with_perm = _build_reference_encoder(latent_dim=16, perm_latents=True)
    assert with_perm.latent_perm is not None
    assert with_perm.latent_perm_inv is not None
    assert with_perm.latent_perm.shape == with_perm.latent_perm_inv.shape
    assert with_perm.latent_perm.shape[0] == with_perm._latent_target_features
    # Round-trip identity ensures packed latents can be permuted and restored losslessly.
    roundtrip = with_perm.latent_perm[with_perm.latent_perm_inv]
    assert torch.equal(roundtrip, torch.arange(with_perm._latent_target_features))

    z_perm = with_perm.encode(x, tau=1.0)
    assert z_perm.shape == (2, 16)
    assert torch.isfinite(z_perm).all()


def test_convpc_posterior_sampling_ctx_is_differentiable() -> None:
    encoder = _build_reference_encoder(latent_dim=16)
    x = torch.randn(3, 1, 32, 32)
    x_flat = encoder._flatten_x(x)

    z, sampling_ctx = encoder._posterior_sample(
        x_flat,
        mpe=False,
        tau=0.7,
        return_sampling_ctx=True,
    )

    assert z.shape == (3, 16)
    assert sampling_ctx.is_differentiable
    assert sampling_ctx.channel_index.is_floating_point()
    assert sampling_ctx.channel_index.dim() == 3
    assert sampling_ctx.repetition_index is not None
    assert sampling_ctx.repetition_index.is_floating_point()


def test_neural_decoder2d_reference_topology_and_parameter_count() -> None:
    decoder = NeuralDecoder2D(
        latent_dim=64,
        output_shape=(1, 32, 32),
        num_hidden=64,
        num_res_hidden=16,
        num_res_layers=2,
        num_scales=2,
        bn=True,
        out_activation="tanh",
    )
    num_params = sum(p.numel() for p in decoder.parameters())
    assert num_params == 357_281

    z = torch.randn(5, 64)
    x = decoder(z)
    assert x.shape == (5, 1, 32, 32)
    assert torch.isfinite(x).all()
