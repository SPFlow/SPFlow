"""Tests for the Conv-PC-based APC joint encoder."""

import pytest
import torch

from spflow.exceptions import InvalidParameterError
from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder


def _build_encoder() -> ConvPcJointEncoder:
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
    )


def test_convpc_apc_encode_decode_and_likelihood_shapes():
    torch.manual_seed(10)
    encoder = _build_encoder()
    x = torch.randn(5, 1, 4, 4)

    z = encoder.encode(x, tau=0.9)
    assert z.shape == (5, 4)
    assert torch.isfinite(z).all()

    x_rec = encoder.decode(z, tau=0.9)
    assert x_rec.shape == (5, 1, 4, 4)
    assert torch.isfinite(x_rec).all()

    ll_joint = encoder.joint_log_likelihood(x, z)
    ll_x = encoder.log_likelihood_x(x)
    # Per-sample vectors are expected by APC objectives that combine these terms.
    assert ll_joint.shape == (5,)
    assert ll_x.shape == (5,)
    assert torch.isfinite(ll_joint).all()
    assert torch.isfinite(ll_x).all()

    z_prior = encoder.sample_prior_z(num_samples=6, tau=0.9)
    assert z_prior.shape == (6, 4)
    assert torch.isfinite(z_prior).all()


def test_convpc_apc_encode_latent_stats_shapes_and_finiteness():
    torch.manual_seed(11)
    encoder = _build_encoder()
    x = torch.randn(4, 1, 4, 4)

    stats, z = encoder.encode(x, return_latent_stats=True)
    assert isinstance(stats, LatentStats)
    assert stats.mu.shape == (4, 4)
    assert stats.logvar.shape == (4, 4)
    assert z.shape == (4, 4)
    assert torch.isfinite(stats.mu).all()
    assert torch.isfinite(stats.logvar).all()
    assert torch.isfinite(z).all()

    stats_direct = encoder.latent_stats(x)
    assert stats_direct.mu.shape == (4, 4)
    assert stats_direct.logvar.shape == (4, 4)
    assert torch.isfinite(stats_direct.mu).all()
    assert torch.isfinite(stats_direct.logvar).all()


def test_convpc_apc_encode_latent_stats_mpe_is_deterministic_and_matches_nonstats() -> None:
    torch.manual_seed(14)
    encoder = _build_encoder()
    x = torch.randn(5, 1, 4, 4)

    stats_first, z_stats_first = encoder.encode(x, mpe=True, tau=0.9, return_latent_stats=True)
    stats_second, z_stats_second = encoder.encode(x, mpe=True, tau=0.9, return_latent_stats=True)
    z_nonstats = encoder.encode(x, mpe=True, tau=0.9, return_latent_stats=False)

    assert isinstance(stats_first, LatentStats)
    assert isinstance(stats_second, LatentStats)
    assert torch.equal(z_stats_first, z_stats_second)
    assert torch.equal(z_stats_first, z_nonstats)
    assert torch.isfinite(stats_first.mu).all()
    assert torch.isfinite(stats_first.logvar).all()
    assert torch.isfinite(stats_second.mu).all()
    assert torch.isfinite(stats_second.logvar).all()


def test_convpc_apc_decode_fill_evidence_keeps_observed_entries():
    torch.manual_seed(12)
    encoder = _build_encoder()
    x = torch.randn(3, 1, 4, 4)
    z = encoder.encode(x, tau=1.0)

    x_partial = x.clone()
    x_partial[:, :, 0, 0] = float("nan")
    x_partial[:, :, 1, 2] = float("nan")

    x_rec = encoder.decode(z, x=x_partial, fill_evidence=True)
    observed = torch.isfinite(x_partial)
    # Evidence fill should be imputation-only; observed values are invariants.
    assert torch.equal(x_rec[observed], x_partial[observed].to(x_rec.dtype))


def test_convpc_reference_allows_latent_dim_mismatch_via_packing():
    # Reference mode packs/unpacks latents, so strict width matching is intentionally relaxed.
    encoder = ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=3,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
    )
    x = torch.randn(2, 1, 4, 4)
    z = encoder.encode(x, tau=1.0)
    assert z.shape == (2, 3)
    assert torch.isfinite(z).all()


def test_convpc_legacy_requires_matching_latent_dim_for_injection_depth():
    # Legacy mode keeps the old strict contract to avoid silently changing historical behavior.
    with pytest.raises(InvalidParameterError, match="latent_dim must match the feature count"):
        ConvPcJointEncoder(
            input_height=4,
            input_width=4,
            input_channels=1,
            latent_dim=3,
            channels=4,
            depth=2,
            kernel_size=2,
            num_repetitions=1,
            use_sum_conv=False,
            latent_depth=0,
            architecture="legacy",
        )
