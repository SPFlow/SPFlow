"""Tests for APC model losses and gradients."""

import torch
from torch.testing import assert_close

from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.decoders import MLPDecoder1D
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


def test_apc_loss_components_are_finite_and_consistent():
    torch.manual_seed(20)
    model = _build_model()
    x = torch.randn(7, 4)
    x[:, 0] = float("nan")

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
