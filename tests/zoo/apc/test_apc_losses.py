"""Tests for APC model losses and gradients."""

import torch
from torch.testing import assert_close

from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.decoders import MLPDecoder1D
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
from spflow.zoo.apc.model import AutoencodingPC


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
        loss_weights=ApcLossWeights(rec=1.0, kld=0.2, nll=0.8),
    )
    return AutoencodingPC(encoder=encoder, decoder=decoder, config=config)


def _build_conv_model() -> AutoencodingPC:
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
        loss_weights=ApcLossWeights(rec=1.0, kld=0.2, nll=0.8),
    )
    return AutoencodingPC(encoder=encoder, decoder=decoder, config=config)


def test_apc_loss_components_are_finite_and_consistent():
    torch.manual_seed(20)
    model = _build_model()
    x = torch.randn(7, 4)

    losses = model.loss_components(x)
    expected_keys = {"rec", "kld", "nll", "total", "z", "x_rec", "mu", "logvar"}
    assert expected_keys.issubset(losses.keys())

    for key in ("rec", "kld", "nll", "total"):
        assert losses[key].shape == torch.Size([])
        assert torch.isfinite(losses[key])

    assert losses["z"].shape == (7, 2)
    assert losses["x_rec"].shape == (7, 4)
    assert losses["mu"].shape == (7, 2)
    assert losses["logvar"].shape == (7, 2)

    weights = model.config.loss_weights
    expected_total = weights.rec * losses["rec"] + weights.kld * losses["kld"] + weights.nll * losses["nll"]
    assert_close(losses["total"], expected_total)


def test_apc_loss_backward_populates_gradients():
    torch.manual_seed(21)
    model = _build_model()
    x = torch.randn(8, 4)

    total = model.loss(x)
    total.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


def _has_finite_nonzero_grad(module: torch.nn.Module) -> bool:
    """Return whether a module has at least one finite, non-zero gradient tensor."""
    for p in module.parameters():
        if p.grad is None:
            continue
        if torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            return True
    return False


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


def test_apc_each_loss_term_has_working_gradients():
    torch.manual_seed(22)
    model = _build_model()
    x = torch.randn(10, 4)

    # Reconstruction loss should train both encoder and decoder.
    model.zero_grad(set_to_none=True)
    rec = model.loss_components(x)["rec"]
    rec.backward()
    assert _has_finite_nonzero_grad(model.encoder)
    assert _has_finite_nonzero_grad(model.decoder)
    assert _has_finite_nonzero_grad_matching_name(model.encoder, "logits")

    # NLL term flows through encoder likelihood only (no decoder path).
    model.zero_grad(set_to_none=True)
    nll = model.loss_components(x)["nll"]
    nll.backward()
    assert _has_finite_nonzero_grad(model.encoder)
    assert not _has_finite_nonzero_grad(model.decoder)

    # KL term uses encoder latent stats only (no decoder path).
    model.zero_grad(set_to_none=True)
    kld = model.loss_components(x)["kld"]
    kld.backward()
    assert _has_finite_nonzero_grad(model.encoder)
    assert not _has_finite_nonzero_grad(model.decoder)


def test_apc_each_loss_term_has_working_gradients_conv_encoder():
    torch.manual_seed(23)
    model = _build_conv_model()
    x = torch.randn(10, 1, 4, 4)

    # Reconstruction loss should train both encoder and decoder.
    model.zero_grad(set_to_none=True)
    rec = model.loss_components(x)["rec"]
    rec.backward()
    assert _has_finite_nonzero_grad(model.encoder)
    assert _has_finite_nonzero_grad(model.decoder)
    assert _has_finite_nonzero_grad_matching_name(model.encoder, "logits")

    # NLL term flows through encoder likelihood only (no decoder path).
    model.zero_grad(set_to_none=True)
    nll = model.loss_components(x)["nll"]
    nll.backward()
    assert _has_finite_nonzero_grad(model.encoder)
    assert not _has_finite_nonzero_grad(model.decoder)

    # KL term uses encoder latent stats only (no decoder path).
    model.zero_grad(set_to_none=True)
    kld = model.loss_components(x)["kld"]
    kld.backward()
    assert _has_finite_nonzero_grad(model.encoder)
    assert not _has_finite_nonzero_grad(model.decoder)
