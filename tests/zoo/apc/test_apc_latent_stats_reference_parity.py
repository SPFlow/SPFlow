"""Regression checks for APC latent-stat rollback behavior."""

import pytest
import torch

from spflow.exceptions import UnsupportedOperationError
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder


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


def test_einet_latent_stats_paths_are_unsupported() -> None:
    encoder = _build_einet_encoder()
    x = torch.randn(8, 4)

    with pytest.raises(UnsupportedOperationError):
        encoder.encode(x, return_latent_stats=True)

    with pytest.raises(UnsupportedOperationError):
        encoder.latent_stats(x)


def test_convpc_latent_stats_paths_are_unsupported() -> None:
    encoder = _build_convpc_encoder()
    x = torch.randn(8, 1, 4, 4)

    with pytest.raises(UnsupportedOperationError):
        encoder.encode(x, return_latent_stats=True)

    with pytest.raises(UnsupportedOperationError):
        encoder.latent_stats(x)
