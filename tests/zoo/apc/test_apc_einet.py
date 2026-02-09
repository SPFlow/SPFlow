"""Tests for the Einet-based APC joint encoder."""

import torch

from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder


def _build_encoder() -> EinetJointEncoder:
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


def test_einet_apc_encode_decode_and_likelihood_shapes():
    torch.manual_seed(0)
    encoder = _build_encoder()
    x = torch.randn(6, 4)

    z = encoder.encode(x, tau=0.7)
    assert z.shape == (6, 2)
    assert torch.isfinite(z).all()

    x_rec = encoder.decode(z, tau=0.7)
    assert x_rec.shape == (6, 4)
    assert torch.isfinite(x_rec).all()

    ll_joint = encoder.joint_log_likelihood(x, z)
    ll_x = encoder.log_likelihood_x(x)
    assert ll_joint.shape == (6,)
    assert ll_x.shape == (6,)
    assert torch.isfinite(ll_joint).all()
    assert torch.isfinite(ll_x).all()

    z_prior = encoder.sample_prior_z(num_samples=5, tau=0.7)
    assert z_prior.shape == (5, 2)
    assert torch.isfinite(z_prior).all()


def test_einet_apc_encode_returns_latent_stats():
    torch.manual_seed(1)
    encoder = _build_encoder()
    x = torch.randn(4, 4)

    stats, z = encoder.encode(x, return_latent_stats=True)
    assert isinstance(stats, LatentStats)
    assert stats.mu.shape == (4, 2)
    assert stats.logvar.shape == (4, 2)
    assert z.shape == (4, 2)
    assert torch.isfinite(stats.mu).all()
    assert torch.isfinite(stats.logvar).all()
    assert torch.isfinite(z).all()


def test_einet_apc_decode_fill_evidence_keeps_observed_entries():
    torch.manual_seed(2)
    encoder = _build_encoder()
    x = torch.randn(5, 4)
    z = encoder.encode(x, tau=1.0)

    x_partial = x.clone()
    x_partial[:, 1] = float("nan")
    x_partial[:, 3] = float("nan")

    x_rec = encoder.decode(z, x=x_partial, fill_evidence=True)
    observed = torch.isfinite(x_partial)
    assert torch.equal(x_rec[observed], x_partial[observed].to(x_rec.dtype))


def test_einet_apc_encode_with_missing_x_values():
    torch.manual_seed(3)
    encoder = _build_encoder()
    x = torch.randn(5, 4)
    x[:, 0] = float("nan")

    z = encoder.encode(x, tau=0.8)
    assert z.shape == (5, 2)
    assert torch.isfinite(z).all()
