"""Tests for APC loss API rollback behavior."""

import pytest
import torch

from spflow.exceptions import UnsupportedOperationError
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


@pytest.mark.parametrize("builder,shape", [(_build_model, (7, 4)), (_build_conv_model, (7, 1, 4, 4))])
def test_apc_loss_components_is_unsupported(builder, shape):
    model = builder()
    x = torch.randn(*shape)

    with pytest.raises(UnsupportedOperationError):
        model.loss_components(x)


@pytest.mark.parametrize("builder,shape", [(_build_model, (8, 4)), (_build_conv_model, (8, 1, 4, 4))])
def test_apc_loss_is_unsupported(builder, shape):
    model = builder()
    x = torch.randn(*shape)

    with pytest.raises(UnsupportedOperationError):
        model.loss(x)
