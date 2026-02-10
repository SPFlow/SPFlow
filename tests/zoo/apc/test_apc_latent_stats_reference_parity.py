"""Reference-parity checks for APC latent stats extraction semantics."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.testing import assert_close

from spflow.modules.leaves import Normal
from spflow.utils.sampling_context import SamplingContext
from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.decoders import MLPDecoder1D
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
from spflow.zoo.apc.model import AutoencodingPC


def _build_einet_encoder() -> EinetJointEncoder:
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


def _build_convpc_encoder() -> ConvPcJointEncoder:
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


def _stamp_distinct_latent_leaf_params(
    encoder: EinetJointEncoder | ConvPcJointEncoder,
) -> None:
    z_leaf = encoder._z_leaf
    if not isinstance(z_leaf, Normal):
        raise AssertionError("Parity tests require a Normal latent leaf.")

    latent_dim, out_channels, repetitions = z_leaf.loc.shape
    f = torch.arange(latent_dim, dtype=z_leaf.loc.dtype, device=z_leaf.loc.device).view(latent_dim, 1, 1)
    c = torch.arange(out_channels, dtype=z_leaf.loc.dtype, device=z_leaf.loc.device).view(1, out_channels, 1)
    r = torch.arange(repetitions, dtype=z_leaf.loc.dtype, device=z_leaf.loc.device).view(1, 1, repetitions)

    loc = 0.25 * f + 0.5 * c + 0.1 * r
    scale = torch.full_like(loc, 1.25) + 0.05 * f + 0.01 * c + 0.005 * r

    with torch.no_grad():
        z_leaf.loc.copy_(loc)
        z_leaf.scale = scale


def _select_latent_channels(
    *,
    encoder: EinetJointEncoder | ConvPcJointEncoder,
    sampling_ctx: SamplingContext,
    max_channel_idx: int,
) -> torch.Tensor:
    channel_index = sampling_ctx.channel_index
    if channel_index.shape[1] == 1:
        channel_index = channel_index.expand(-1, encoder.latent_dim)
    elif channel_index.shape[1] == encoder.latent_dim:
        pass
    elif channel_index.shape[1] == (encoder.num_x_features + encoder.latent_dim):
        channel_index = channel_index[:, encoder._z_cols]
    else:
        raise AssertionError(f"Unexpected channel-index width: {channel_index.shape[1]}.")
    return channel_index.to(dtype=torch.long).clamp(min=0, max=max_channel_idx)


def _select_latent_repetitions(
    *,
    encoder: EinetJointEncoder | ConvPcJointEncoder,
    sampling_ctx: SamplingContext,
    batch_size: int,
    max_repetition_idx: int,
) -> torch.Tensor:
    repetition_idx = sampling_ctx.repetition_idx
    if repetition_idx is None:
        return torch.zeros(
            (batch_size, encoder.latent_dim), dtype=torch.long, device=encoder._z_leaf.loc.device
        )

    if repetition_idx.dim() == 1:
        repetition_idx = repetition_idx.unsqueeze(1).expand(-1, encoder.latent_dim)
    elif repetition_idx.shape[1] == 1:
        repetition_idx = repetition_idx.expand(-1, encoder.latent_dim)
    elif repetition_idx.shape[1] == encoder.latent_dim:
        pass
    elif repetition_idx.shape[1] == (encoder.num_x_features + encoder.latent_dim):
        repetition_idx = repetition_idx[:, encoder._z_cols]
    else:
        repetition_idx = repetition_idx[:, :1].expand(-1, encoder.latent_dim)

    return repetition_idx.to(dtype=torch.long).clamp(min=0, max=max_repetition_idx)


def _expected_stats_from_context(
    encoder: EinetJointEncoder | ConvPcJointEncoder,
    sampling_ctx: SamplingContext,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_leaf = encoder._z_leaf
    if not isinstance(z_leaf, Normal):
        raise AssertionError("Parity tests require a Normal latent leaf.")

    loc = z_leaf.loc
    scale = z_leaf.scale.clamp_min(encoder.posterior_var_floor**0.5)

    if isinstance(encoder, ConvPcJointEncoder) and encoder._last_latent_leaf_channel_index is not None:
        channel_idx = encoder._last_latent_leaf_channel_index.to(dtype=torch.long)
        if channel_idx.shape[1] == 1:
            channel_idx = channel_idx.expand(-1, encoder.latent_dim)
        else:
            channel_idx = channel_idx[:, : encoder.latent_dim]
        channel_idx = channel_idx.clamp(min=0, max=loc.shape[1] - 1)

        if encoder._last_latent_leaf_repetition_index is None:
            repetition_idx = torch.zeros(
                (batch_size, encoder.latent_dim),
                dtype=torch.long,
                device=encoder._z_leaf.loc.device,
            )
        else:
            repetition_idx = encoder._last_latent_leaf_repetition_index.to(dtype=torch.long)
            if repetition_idx.shape[1] == 1:
                repetition_idx = repetition_idx.expand(-1, encoder.latent_dim)
            else:
                repetition_idx = repetition_idx[:, : encoder.latent_dim]
            repetition_idx = repetition_idx.clamp(min=0, max=loc.shape[2] - 1)
    else:
        channel_idx = _select_latent_channels(
            encoder=encoder, sampling_ctx=sampling_ctx, max_channel_idx=loc.shape[1] - 1
        )
        repetition_idx = _select_latent_repetitions(
            encoder=encoder,
            sampling_ctx=sampling_ctx,
            batch_size=batch_size,
            max_repetition_idx=loc.shape[2] - 1,
        )

    feat_idx = torch.arange(encoder.latent_dim, device=loc.device).unsqueeze(0).expand(batch_size, -1)
    mu = loc[feat_idx, channel_idx, repetition_idx]
    logvar = (
        (scale[feat_idx, channel_idx, repetition_idx].pow(2)).clamp_min(encoder.posterior_var_floor).log()
    )
    return mu, logvar


def _inject_one_hot_selectors_from_indices(
    encoder: EinetJointEncoder | ConvPcJointEncoder,
    sampling_ctx: SamplingContext,
) -> None:
    """Populate selector tensors from hard routing indices for parity checks."""
    z_leaf = encoder._z_leaf
    if not isinstance(z_leaf, Normal):
        raise AssertionError("Parity tests require a Normal latent leaf.")

    channel_idx = sampling_ctx.channel_index.to(dtype=torch.long).clamp(min=0, max=z_leaf.loc.shape[1] - 1)
    sampling_ctx.channel_select = F.one_hot(channel_idx, num_classes=z_leaf.loc.shape[1]).to(
        dtype=z_leaf.loc.dtype
    )

    repetition_idx = sampling_ctx.repetition_idx
    if repetition_idx is None:
        sampling_ctx.repetition_select = None
        return

    if repetition_idx.dim() == 1:
        repetition_idx = repetition_idx.unsqueeze(1)
    repetition_idx = repetition_idx.to(dtype=torch.long).clamp(min=0, max=z_leaf.loc.shape[2] - 1)
    sampling_ctx.repetition_select = F.one_hot(repetition_idx, num_classes=z_leaf.loc.shape[2]).to(
        dtype=z_leaf.loc.dtype
    )


def _has_finite_nonzero_grad_matching_name(module: torch.nn.Module, name_substring: str) -> bool:
    """Return whether any parameter whose name contains substring has finite non-zero grad."""
    for name, param in module.named_parameters():
        if name_substring not in name:
            continue
        if param.grad is None:
            continue
        if torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0:
            return True
    return False


def test_einet_encode_return_latent_stats_shapes_and_finite() -> None:
    torch.manual_seed(110)
    encoder = _build_einet_encoder()
    x = torch.randn(8, 4)

    stats, z = encoder.encode(x, return_latent_stats=True)
    assert stats.mu.shape == z.shape
    assert stats.logvar.shape == z.shape
    assert torch.isfinite(stats.mu).all()
    assert torch.isfinite(stats.logvar).all()
    assert torch.isfinite(z).all()


def test_convpc_encode_return_latent_stats_shapes_and_finite() -> None:
    torch.manual_seed(111)
    encoder = _build_convpc_encoder()
    x = torch.randn(8, 1, 4, 4)

    stats, z = encoder.encode(x, return_latent_stats=True)
    assert stats.mu.shape == z.shape
    assert stats.logvar.shape == z.shape
    assert torch.isfinite(stats.mu).all()
    assert torch.isfinite(stats.logvar).all()
    assert torch.isfinite(z).all()


def test_einet_latent_stats_use_sampling_context_selected_leaf_params() -> None:
    torch.manual_seed(120)
    encoder = _build_einet_encoder()
    _stamp_distinct_latent_leaf_params(encoder)
    x = torch.randn(5, 4)
    x_flat = encoder._flatten_x(x)

    torch.manual_seed(121)
    z_ref, sampling_ctx = encoder._posterior_sample(x_flat, mpe=False, tau=0.9, return_sampling_ctx=True)
    expected_mu, expected_logvar = _expected_stats_from_context(
        encoder=encoder, sampling_ctx=sampling_ctx, batch_size=z_ref.shape[0]
    )
    expected_z = 0.9 * torch.randn_like(expected_mu) * torch.exp(0.5 * expected_logvar) + expected_mu

    torch.manual_seed(121)
    stats, z = encoder.encode(x, mpe=False, tau=0.9, return_latent_stats=True)

    assert_close(stats.mu, expected_mu, rtol=0.0, atol=1e-6)
    assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=1e-6)
    assert_close(z, expected_z, rtol=0.0, atol=1e-6)


def test_einet_latent_stats_match_hard_indices_with_one_hot_selectors() -> None:
    torch.manual_seed(125)
    encoder = _build_einet_encoder()
    _stamp_distinct_latent_leaf_params(encoder)
    x = torch.randn(5, 4)
    x_flat = encoder._flatten_x(x)

    _, sampling_ctx = encoder._posterior_sample(x_flat, mpe=False, tau=0.9, return_sampling_ctx=True)
    _inject_one_hot_selectors_from_indices(encoder, sampling_ctx)

    expected_mu, expected_logvar = _expected_stats_from_context(
        encoder=encoder, sampling_ctx=sampling_ctx, batch_size=x.shape[0]
    )
    stats = encoder._latent_stats_from_leaf_params(sampling_ctx=sampling_ctx, batch_size=x.shape[0])
    assert_close(stats.mu, expected_mu, rtol=0.0, atol=1e-6)
    assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=1e-6)


def test_convpc_latent_stats_use_sampling_context_selected_leaf_params() -> None:
    torch.manual_seed(130)
    encoder = _build_convpc_encoder()
    _stamp_distinct_latent_leaf_params(encoder)
    x = torch.randn(5, 1, 4, 4)
    x_flat = encoder._flatten_x(x)

    torch.manual_seed(131)
    z_ref, sampling_ctx = encoder._posterior_sample(x_flat, mpe=False, tau=0.9, return_sampling_ctx=True)
    expected_mu, expected_logvar = _expected_stats_from_context(
        encoder=encoder, sampling_ctx=sampling_ctx, batch_size=z_ref.shape[0]
    )
    expected_z = 0.9 * torch.randn_like(expected_mu) * torch.exp(0.5 * expected_logvar) + expected_mu

    torch.manual_seed(131)
    stats, z = encoder.encode(x, mpe=False, tau=0.9, return_latent_stats=True)

    assert_close(stats.mu, expected_mu, rtol=0.0, atol=1e-6)
    assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=1e-6)
    assert_close(z, expected_z, rtol=0.0, atol=1e-6)


def test_convpc_latent_stats_match_hard_indices_with_one_hot_selectors() -> None:
    torch.manual_seed(135)
    encoder = _build_convpc_encoder()
    _stamp_distinct_latent_leaf_params(encoder)
    x = torch.randn(5, 1, 4, 4)
    x_flat = encoder._flatten_x(x)

    _, sampling_ctx = encoder._posterior_sample(x_flat, mpe=False, tau=0.9, return_sampling_ctx=True)
    _inject_one_hot_selectors_from_indices(encoder, sampling_ctx)
    # Force selector resolution from sampling context for this parity check.
    encoder._last_latent_leaf_channel_index = None
    encoder._last_latent_leaf_repetition_index = None
    encoder._last_latent_leaf_channel_select = None
    encoder._last_latent_leaf_repetition_select = None

    expected_mu, expected_logvar = _expected_stats_from_context(
        encoder=encoder, sampling_ctx=sampling_ctx, batch_size=x.shape[0]
    )
    stats = encoder._latent_stats_from_leaf_params(sampling_ctx=sampling_ctx, batch_size=x.shape[0])
    assert_close(stats.mu, expected_mu, rtol=0.0, atol=1e-6)
    assert_close(stats.logvar, expected_logvar, rtol=0.0, atol=1e-6)


def test_einet_reparameterization_contract_is_finite_with_plausible_scale() -> None:
    torch.manual_seed(140)
    encoder = _build_einet_encoder()
    _stamp_distinct_latent_leaf_params(encoder)
    x = torch.randn(64, 4)

    stats, z = encoder.encode(x, mpe=False, tau=1.0, return_latent_stats=True)
    eps = (z - stats.mu) / torch.exp(0.5 * stats.logvar)
    eps = eps.detach()

    assert torch.isfinite(eps).all()
    assert eps.abs().mean().item() < 5.0


def test_convpc_reparameterization_contract_is_finite_with_plausible_scale() -> None:
    torch.manual_seed(141)
    encoder = _build_convpc_encoder()
    _stamp_distinct_latent_leaf_params(encoder)
    x = torch.randn(64, 1, 4, 4)

    stats, z = encoder.encode(x, mpe=False, tau=1.0, return_latent_stats=True)
    eps = (z - stats.mu) / torch.exp(0.5 * stats.logvar)
    eps = eps.detach()

    assert torch.isfinite(eps).all()
    assert eps.abs().mean().item() < 5.0


def test_einet_return_latent_stats_path_propagates_rec_grad_to_routing_logits() -> None:
    torch.manual_seed(150)
    model = AutoencodingPC(
        encoder=_build_einet_encoder(),
        decoder=MLPDecoder1D(latent_dim=2, output_dim=4, hidden_dims=(16,)),
        config=ApcConfig(
            latent_dim=2,
            rec_loss="mse",
            sample_tau=1.0,
            loss_weights=ApcLossWeights(rec=1.0, kld=0.2, nll=0.8),
        ),
    )
    x = torch.randn(12, 4)
    model.zero_grad(set_to_none=True)
    model.loss_components(x)["rec"].backward()
    assert _has_finite_nonzero_grad_matching_name(model.encoder, "logits")


def test_convpc_return_latent_stats_path_propagates_rec_grad_to_routing_logits() -> None:
    torch.manual_seed(151)
    model = AutoencodingPC(
        encoder=_build_convpc_encoder(),
        decoder=MLPDecoder1D(latent_dim=4, output_dim=16, hidden_dims=(16,)),
        config=ApcConfig(
            latent_dim=4,
            rec_loss="mse",
            sample_tau=1.0,
            loss_weights=ApcLossWeights(rec=1.0, kld=0.2, nll=0.8),
        ),
    )
    x = torch.randn(12, 1, 4, 4)
    model.zero_grad(set_to_none=True)
    model.loss_components(x)["rec"].backward()
    assert _has_finite_nonzero_grad_matching_name(model.encoder, "logits")
