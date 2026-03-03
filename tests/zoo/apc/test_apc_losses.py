"""Tests for APC loss objective behavior."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from spflow.exceptions import UnsupportedOperationError
from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.decoders import MLPDecoder1D
from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
from spflow.zoo.apc.model import AutoencodingPC


class _ToyStatsEncoder(nn.Module):
    """Minimal encoder stub that supports latent stats for objective tests."""

    num_x_features: int = 2
    latent_dim: int = 2

    def encode(
        self,
        x: Tensor,
        *,
        mpe: bool = False,
        tau: float = 1.0,
        return_latent_stats: bool = False,
    ) -> Tensor | tuple[LatentStats, Tensor]:
        del mpe, tau
        x_flat = x.reshape(x.shape[0], -1).to(dtype=torch.get_default_dtype())
        z_samples = x_flat[:, : self.latent_dim] + 0.25
        if not return_latent_stats:
            return z_samples
        mu = torch.full_like(z_samples, 1.5)
        logvar = torch.full_like(z_samples, -0.2)
        return LatentStats(mu=mu, logvar=logvar), z_samples

    def decode(
        self,
        z: Tensor,
        *,
        x: Tensor | None = None,
        mpe: bool = False,
        tau: float = 1.0,
        fill_evidence: bool = False,
    ) -> Tensor:
        del mpe, tau
        z_flat = z.reshape(z.shape[0], -1)
        if x is None or not fill_evidence:
            return z_flat
        x_flat = x.reshape(x.shape[0], -1).to(dtype=z_flat.dtype)
        finite_mask = torch.isfinite(x_flat)
        return torch.where(finite_mask, x_flat, z_flat)

    def joint_log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        x_flat = x.reshape(x.shape[0], -1).to(dtype=z.dtype)
        z_flat = z.reshape(z.shape[0], -1)
        return -((x_flat - z_flat) ** 2).sum(dim=1)

    def log_likelihood_x(self, x: Tensor) -> Tensor:
        x_flat = x.reshape(x.shape[0], -1).to(dtype=torch.get_default_dtype())
        return -(x_flat**2).sum(dim=1)

    def sample_prior_z(self, num_samples: int, *, tau: float = 1.0) -> Tensor:
        del tau
        return torch.zeros((num_samples, self.latent_dim), dtype=torch.get_default_dtype())

    def latent_stats(self, x: Tensor, *, tau: float = 1.0) -> LatentStats:
        del x, tau
        mu = torch.ones((1, self.latent_dim), dtype=torch.get_default_dtype())
        logvar = torch.zeros_like(mu)
        return LatentStats(mu=mu, logvar=logvar)


class _RecordingDecoder(nn.Module):
    """Identity decoder that records the latent input passed by APC."""

    def __init__(self) -> None:
        super().__init__()
        self.last_input: Tensor | None = None

    def forward(self, z: Tensor) -> Tensor:
        self.last_input = z.detach().clone()
        return z


class _NoStatsEncoder(_ToyStatsEncoder):
    """Encoder stub that does not honor return_latent_stats contract."""

    def encode(
        self,
        x: Tensor,
        *,
        mpe: bool = False,
        tau: float = 1.0,
        return_latent_stats: bool = False,
    ) -> Tensor | tuple[LatentStats, Tensor]:
        del return_latent_stats
        return super().encode(x, mpe=mpe, tau=tau, return_latent_stats=False)


def _build_einet_model(
    *, weights: ApcLossWeights, nll_x_and_z: bool = True, train_decode_mpe: bool = False
) -> AutoencodingPC:
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
        nll_x_and_z=nll_x_and_z,
        train_decode_mpe=train_decode_mpe,
        loss_weights=weights,
    )
    return AutoencodingPC(encoder=encoder, decoder=decoder, config=config)


def _build_conv_model(
    *, weights: ApcLossWeights, nll_x_and_z: bool = True, train_decode_mpe: bool = False
) -> AutoencodingPC:
    encoder = ConvPcJointEncoder(
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
    decoder = MLPDecoder1D(latent_dim=4, output_dim=16, hidden_dims=(16,))
    config = ApcConfig(
        latent_dim=4,
        rec_loss="mse",
        sample_tau=1.0,
        nll_x_and_z=nll_x_and_z,
        train_decode_mpe=train_decode_mpe,
        loss_weights=weights,
    )
    return AutoencodingPC(encoder=encoder, decoder=decoder, config=config)


def test_apc_loss_components_returns_terms_and_weighted_total() -> None:
    encoder = _ToyStatsEncoder()
    decoder = _RecordingDecoder()
    config = ApcConfig(
        latent_dim=2,
        rec_loss="mse",
        sample_tau=1.0,
        nll_x_and_z=True,
        train_decode_mpe=False,
        loss_weights=ApcLossWeights(rec=1.1, kld=0.4, nll=0.7),
    )
    model = AutoencodingPC(encoder=encoder, decoder=decoder, config=config)
    x = torch.randn(6, 2)

    losses = model.loss_components(x)

    for key in ("rec", "kld", "nll", "total"):
        assert key in losses
        assert losses[key].shape == ()
        assert torch.isfinite(losses[key])

    expected_total = (
        config.loss_weights.rec * losses["rec"]
        + config.loss_weights.kld * losses["kld"]
        + config.loss_weights.nll * losses["nll"]
    )
    torch.testing.assert_close(losses["total"], expected_total, rtol=0.0, atol=1e-8)
    torch.testing.assert_close(model.loss(x), losses["total"], rtol=0.0, atol=1e-8)


@pytest.mark.parametrize(
    "nll_x_and_z,expected_nll,expected_calls", [(True, 3.0, (1, 0)), (False, 5.0, (0, 1))]
)
def test_apc_loss_components_switches_nll_mode(
    monkeypatch: pytest.MonkeyPatch,
    nll_x_and_z: bool,
    expected_nll: float,
    expected_calls: tuple[int, int],
) -> None:
    model = AutoencodingPC(
        encoder=_ToyStatsEncoder(),
        decoder=_RecordingDecoder(),
        config=ApcConfig(
            latent_dim=2,
            rec_loss="mse",
            sample_tau=1.0,
            nll_x_and_z=nll_x_and_z,
            loss_weights=ApcLossWeights(rec=0.0, kld=0.0, nll=1.0),
        ),
    )
    x = torch.randn(5, 2)
    calls = {"joint": 0, "marginal": 0}

    def _joint(_: Tensor, z: Tensor) -> Tensor:
        calls["joint"] += 1
        return z.new_full((z.shape[0],), -3.0)

    def _marginal(x_in: Tensor) -> Tensor:
        calls["marginal"] += 1
        return x_in.new_full((x_in.shape[0],), -5.0)

    monkeypatch.setattr(model, "joint_log_likelihood", _joint)
    monkeypatch.setattr(model, "log_likelihood_x", _marginal)

    losses = model.loss_components(x)
    torch.testing.assert_close(
        losses["nll"],
        torch.tensor(expected_nll, dtype=losses["nll"].dtype, device=losses["nll"].device),
        rtol=0.0,
        atol=1e-8,
    )
    assert calls["joint"] == expected_calls[0]
    assert calls["marginal"] == expected_calls[1]


def test_apc_loss_components_train_decode_mpe_uses_posterior_mean() -> None:
    decoder = _RecordingDecoder()
    model = AutoencodingPC(
        encoder=_ToyStatsEncoder(),
        decoder=decoder,
        config=ApcConfig(
            latent_dim=2,
            rec_loss="mse",
            sample_tau=1.0,
            train_decode_mpe=True,
            loss_weights=ApcLossWeights(rec=1.0, kld=0.0, nll=0.0),
        ),
    )
    x = torch.randn(4, 2)

    losses = model.loss_components(x)

    assert decoder.last_input is not None
    assert "mu" in losses
    torch.testing.assert_close(decoder.last_input, losses["mu"], rtol=0.0, atol=0.0)


def test_apc_loss_components_fails_fast_kl_without_latent_stats() -> None:
    model = AutoencodingPC(
        encoder=_NoStatsEncoder(),
        decoder=_RecordingDecoder(),
        config=ApcConfig(
            latent_dim=2,
            rec_loss="mse",
            sample_tau=1.0,
            nll_x_and_z=True,
            train_decode_mpe=False,
            loss_weights=ApcLossWeights(rec=0.0, kld=1.0, nll=0.0),
        ),
    )
    x = torch.randn(8, 4)

    with pytest.raises(UnsupportedOperationError):
        model.loss_components(x)


@pytest.mark.parametrize(
    "builder,shape",
    [
        (_build_einet_model, (7, 4)),
        (_build_conv_model, (7, 1, 4, 4)),
    ],
)
def test_apc_loss_components_supports_current_encoders_when_kl_disabled(builder, shape) -> None:
    model = builder(weights=ApcLossWeights(rec=1.0, kld=0.0, nll=0.8))
    x = torch.randn(*shape)

    losses = model.loss_components(x)
    assert torch.isfinite(losses["rec"])
    assert torch.isfinite(losses["nll"])
    assert torch.isfinite(losses["total"])
    torch.testing.assert_close(
        losses["kld"],
        torch.zeros((), dtype=losses["kld"].dtype, device=losses["kld"].device),
        rtol=0.0,
        atol=0.0,
    )
