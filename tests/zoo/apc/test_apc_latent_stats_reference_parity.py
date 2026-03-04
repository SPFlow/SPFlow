"""Regression checks for APC latent-stat extraction from latent leaf parameters."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spflow.exceptions import UnsupportedOperationError
from spflow.modules.leaves import Bernoulli, Binomial, Categorical, Normal, Poisson
from spflow.modules.leaves.leaf import LeafModule
from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
from spflow.zoo.apc.model import AutoencodingPC


def _fixed_normal_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    num_features = len(scope_indices)
    loc = torch.linspace(-0.6, 0.6, steps=num_features).reshape(num_features, 1, 1)
    loc = loc.expand(num_features, out_channels, num_repetitions).clone()
    logvar = torch.linspace(-1.4, -0.2, steps=num_features).reshape(num_features, 1, 1)
    logvar = logvar.expand(num_features, out_channels, num_repetitions).clone()
    scale = torch.exp(0.5 * logvar)
    return Normal(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
        loc=loc,
        scale=scale,
    )


def _bernoulli_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    probs = torch.full((len(scope_indices), out_channels, num_repetitions), 0.35)
    return Bernoulli(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
        probs=probs,
    )


def _binomial_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    total_count = torch.full((len(scope_indices), out_channels, num_repetitions), 6.0)
    probs = torch.full((len(scope_indices), out_channels, num_repetitions), 0.3)
    return Binomial(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
        total_count=total_count,
        probs=probs,
    )


def _categorical_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    logits = torch.tensor([1.2, -0.4, 0.1, 0.5], dtype=torch.get_default_dtype())
    logits = logits.view(1, 1, 1, 4).expand(len(scope_indices), out_channels, num_repetitions, 4).clone()
    return Categorical(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
        logits=logits,
    )


def _poisson_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    rate = torch.full((len(scope_indices), out_channels, num_repetitions), 3.0)
    return Poisson(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
        rate=rate,
    )


def _build_einet_encoder(*, latent_dim: int, z_leaf_factory) -> EinetJointEncoder:
    return EinetJointEncoder(
        num_x_features=4,
        latent_dim=latent_dim,
        num_sums=4,
        num_leaves=1,
        depth=1,
        num_repetitions=1,
        layer_type="linsum",
        z_leaf_factory=z_leaf_factory,
    )


def _build_conv_encoder(*, latent_dim: int, z_leaf_factory) -> ConvPcJointEncoder:
    return ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=latent_dim,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
        latent_channels=1,
        z_leaf_factory=z_leaf_factory,
    )


def _assert_has_finite_nonzero_encoder_grads(encoder: nn.Module) -> None:
    grads = [param.grad for param in encoder.parameters() if param.requires_grad and param.grad is not None]
    assert grads, "Expected at least one encoder parameter gradient."
    for grad in grads:
        assert torch.isfinite(grad).all()
    total_abs_grad = torch.stack([grad.abs().sum() for grad in grads]).sum()
    assert float(total_abs_grad.item()) > 0.0


def test_einet_latent_stats_match_normal_leaf_params_for_single_component() -> None:
    torch.manual_seed(101)
    encoder = _build_einet_encoder(latent_dim=2, z_leaf_factory=_fixed_normal_leaf)
    x = torch.randn(6, 4)

    stats, z = encoder.encode(x, mpe=False, tau=0.8, return_latent_stats=True)
    assert z.shape == (6, 2)
    expected_mu = encoder._z_leaf.loc[:, 0, 0].unsqueeze(0).expand(6, -1)
    expected_logvar = (2.0 * encoder._z_leaf.log_scale[:, 0, 0]).unsqueeze(0).expand(6, -1)
    expected_kld = 0.5 * (expected_mu.pow(2) + expected_logvar.exp() - 1.0 - expected_logvar).sum(dim=1)
    torch.testing.assert_close(stats.mu, expected_mu, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stats.kld_per_sample, expected_kld, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.decode_latent, expected_mu, rtol=0.0, atol=0.0)

    stats_direct = encoder.latent_stats(x, tau=0.8)
    torch.testing.assert_close(stats_direct.mu, expected_mu, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stats_direct.logvar, expected_logvar, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stats_direct.kld_per_sample, expected_kld, rtol=0.0, atol=1e-6)


def test_convpc_latent_stats_match_normal_leaf_params_for_single_component() -> None:
    torch.manual_seed(102)
    encoder = _build_conv_encoder(latent_dim=4, z_leaf_factory=_fixed_normal_leaf)
    x = torch.randn(5, 1, 4, 4)

    stats, z = encoder.encode(x, mpe=False, tau=0.8, return_latent_stats=True)
    assert z.shape == (5, 4)
    expected_mu = encoder._z_leaf.loc[:, 0, 0].unsqueeze(0).expand(5, -1)
    expected_logvar = (2.0 * encoder._z_leaf.log_scale[:, 0, 0]).unsqueeze(0).expand(5, -1)
    expected_kld = 0.5 * (expected_mu.pow(2) + expected_logvar.exp() - 1.0 - expected_logvar).sum(dim=1)
    torch.testing.assert_close(stats.mu, expected_mu, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stats.kld_per_sample, expected_kld, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.decode_latent, expected_mu, rtol=0.0, atol=0.0)

    stats_direct = encoder.latent_stats(x, tau=0.8)
    torch.testing.assert_close(stats_direct.mu, expected_mu, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stats_direct.logvar, expected_logvar, rtol=0.0, atol=0.0)
    torch.testing.assert_close(stats_direct.kld_per_sample, expected_kld, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    "build_encoder,x_builder",
    [
        (
            lambda: _build_einet_encoder(latent_dim=2, z_leaf_factory=_bernoulli_leaf),
            lambda: torch.randn(7, 4),
        ),
        (
            lambda: _build_conv_encoder(latent_dim=2, z_leaf_factory=_bernoulli_leaf),
            lambda: torch.randn(7, 1, 4, 4),
        ),
    ],
)
def test_non_normal_bernoulli_latent_stats_are_exact_and_return_discrete_z(build_encoder, x_builder) -> None:
    torch.manual_seed(103)
    encoder = build_encoder()
    x = x_builder()
    latent_dim = encoder.latent_dim

    stats, z = encoder.encode(x, mpe=False, tau=0.9, return_latent_stats=True)
    p = torch.full((x.shape[0], latent_dim), 0.35, dtype=stats.mu.dtype, device=stats.mu.device)
    one_minus_p = 1.0 - p
    expected_mu = p
    expected_logvar = (p * one_minus_p).log()
    expected_kld = (p * torch.log(p / 0.5) + one_minus_p * torch.log(one_minus_p / 0.5)).sum(dim=1)
    expected_mode = torch.zeros_like(expected_mu)

    torch.testing.assert_close(stats.mu, expected_mu, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.kld_per_sample, expected_kld, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.decode_latent, expected_mode, rtol=0.0, atol=0.0)

    # Non-normal encode(..., return_latent_stats=True, mpe=False) returns traversal samples.
    assert torch.all((z == 0.0) | (z == 1.0))


def test_einet_binomial_latent_stats_are_exact_and_return_discrete_z() -> None:
    torch.manual_seed(104)
    encoder = _build_einet_encoder(latent_dim=2, z_leaf_factory=_binomial_leaf)
    x = torch.randn(6, 4)

    stats, z = encoder.encode(x, mpe=False, tau=0.9, return_latent_stats=True)
    n = torch.full((x.shape[0], 2), 6.0, dtype=stats.mu.dtype, device=stats.mu.device)
    p = torch.full((x.shape[0], 2), 0.3, dtype=stats.mu.dtype, device=stats.mu.device)
    one_minus_p = 1.0 - p
    expected_mu = n * p
    expected_logvar = (n * p * one_minus_p).log()
    expected_kld = (n * (p * torch.log(p / 0.5) + one_minus_p * torch.log(one_minus_p / 0.5))).sum(dim=1)
    expected_mode = torch.floor((n + 1.0) * p).clamp(min=0.0)
    expected_mode = torch.minimum(expected_mode, n)

    torch.testing.assert_close(stats.mu, expected_mu, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.kld_per_sample, expected_kld, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.decode_latent, expected_mode, rtol=0.0, atol=0.0)

    assert torch.all((z >= 0.0) & (z <= 6.0))
    assert torch.allclose(z, torch.round(z), atol=0.0, rtol=0.0)


def test_einet_categorical_latent_stats_are_exact_and_return_discrete_z() -> None:
    torch.manual_seed(105)
    encoder = _build_einet_encoder(latent_dim=2, z_leaf_factory=_categorical_leaf)
    x = torch.randn(8, 4)

    stats, z = encoder.encode(x, mpe=False, tau=0.9, return_latent_stats=True)
    logits = torch.tensor([1.2, -0.4, 0.1, 0.5], dtype=stats.mu.dtype, device=stats.mu.device)
    probs = torch.softmax(logits, dim=0)
    probs = probs.view(1, 1, -1).expand(x.shape[0], 2, -1)
    k_vals = torch.arange(4, dtype=stats.mu.dtype, device=stats.mu.device).view(1, 1, -1)
    expected_mu = (probs * k_vals).sum(dim=-1)
    expected_second = (probs * (k_vals.pow(2))).sum(dim=-1)
    expected_logvar = (expected_second - expected_mu.pow(2)).log()
    expected_kld = (
        (probs * (probs.log() + torch.log(torch.tensor(4.0, device=probs.device)))).sum(dim=-1).sum(dim=1)
    )
    expected_mode = torch.argmax(probs, dim=-1).to(dtype=stats.mu.dtype)

    torch.testing.assert_close(stats.mu, expected_mu, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.kld_per_sample, expected_kld, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stats.decode_latent, expected_mode, rtol=0.0, atol=0.0)

    assert torch.all((z >= 0.0) & (z <= 3.0))
    assert torch.allclose(z, torch.round(z), atol=0.0, rtol=0.0)


def test_train_decode_mpe_uses_decode_latent_for_non_normal_latents() -> None:
    class _RecordingDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_input: torch.Tensor | None = None

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            self.last_input = z.detach().clone()
            return torch.cat([z, z], dim=1)

    torch.manual_seed(106)
    encoder = _build_einet_encoder(latent_dim=2, z_leaf_factory=_bernoulli_leaf)
    decoder = _RecordingDecoder()
    model = AutoencodingPC(
        encoder=encoder,
        decoder=decoder,
        config=ApcConfig(
            latent_dim=2,
            rec_loss="mse",
            sample_tau=1.0,
            train_decode_mpe=True,
            loss_weights=ApcLossWeights(rec=1.0, kld=0.0, nll=0.0),
        ),
    )
    x = torch.randn(5, 4)
    losses = model.loss_components(x)
    assert decoder.last_input is not None
    # Bernoulli decode latent uses mode -> binary.
    assert torch.all((decoder.last_input == 0.0) | (decoder.last_input == 1.0))
    # Mean stats for Bernoulli are probabilities, not binary.
    assert not torch.equal(decoder.last_input, losses["mu"])


def test_unsupported_latent_family_fails_fast_without_fallback() -> None:
    torch.manual_seed(107)
    encoder = _build_einet_encoder(latent_dim=2, z_leaf_factory=_poisson_leaf)
    x = torch.randn(6, 4)

    with pytest.raises(UnsupportedOperationError):
        encoder.encode(x, return_latent_stats=True)

    with pytest.raises(UnsupportedOperationError):
        encoder.latent_stats(x)

    model = AutoencodingPC(
        encoder=encoder,
        decoder=None,
        config=ApcConfig(
            latent_dim=2,
            rec_loss="mse",
            sample_tau=1.0,
            loss_weights=ApcLossWeights(rec=0.0, kld=1.0, nll=0.0),
        ),
    )
    with pytest.raises(UnsupportedOperationError):
        model.loss_components(x)


def test_einet_latent_stats_are_differentiable_wrt_encoder_parameters() -> None:
    torch.manual_seed(108)
    encoder = _build_einet_encoder(latent_dim=2, z_leaf_factory=_fixed_normal_leaf)
    x = torch.randn(8, 4)

    encoder.zero_grad(set_to_none=True)
    stats_mu, _ = encoder.encode(x, mpe=False, tau=0.8, return_latent_stats=True)
    stats_mu.mu.mean().backward()
    _assert_has_finite_nonzero_encoder_grads(encoder)
    assert encoder._z_leaf.loc.grad is not None
    assert torch.isfinite(encoder._z_leaf.loc.grad).all()
    assert float(encoder._z_leaf.loc.grad.abs().sum().item()) > 0.0

    encoder.zero_grad(set_to_none=True)
    stats_logvar, _ = encoder.encode(x, mpe=False, tau=0.8, return_latent_stats=True)
    stats_logvar.logvar.mean().backward()
    _assert_has_finite_nonzero_encoder_grads(encoder)
    assert encoder._z_leaf.log_scale.grad is not None
    assert torch.isfinite(encoder._z_leaf.log_scale.grad).all()
    assert float(encoder._z_leaf.log_scale.grad.abs().sum().item()) > 0.0


def test_convpc_latent_stats_are_differentiable_wrt_encoder_parameters() -> None:
    torch.manual_seed(109)
    encoder = _build_conv_encoder(latent_dim=4, z_leaf_factory=_fixed_normal_leaf)
    x = torch.randn(8, 1, 4, 4)

    encoder.zero_grad(set_to_none=True)
    stats_mu, _ = encoder.encode(x, mpe=False, tau=0.8, return_latent_stats=True)
    stats_mu.mu.mean().backward()
    _assert_has_finite_nonzero_encoder_grads(encoder)
    assert encoder._z_leaf.loc.grad is not None
    assert torch.isfinite(encoder._z_leaf.loc.grad).all()
    assert float(encoder._z_leaf.loc.grad.abs().sum().item()) > 0.0

    encoder.zero_grad(set_to_none=True)
    stats_logvar, _ = encoder.encode(x, mpe=False, tau=0.8, return_latent_stats=True)
    stats_logvar.logvar.mean().backward()
    _assert_has_finite_nonzero_encoder_grads(encoder)
    assert encoder._z_leaf.log_scale.grad is not None
    assert torch.isfinite(encoder._z_leaf.log_scale.grad).all()
    assert float(encoder._z_leaf.log_scale.grad.abs().sum().item()) > 0.0
