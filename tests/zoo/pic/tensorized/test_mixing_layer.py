"""Tests for mixing sum layer."""

import pytest
import torch

from spflow.zoo.pic.tensorized.mixing import MixingSumLayer
from spflow.zoo.pic.tensorized.utils import eval_mixing


class TestMixingSumLayerInit:
    """Tests for MixingSumLayer initialization."""

    def test_rejects_mismatched_input_output_units(self):
        """Mixing layer requires equal input/output unit counts."""
        with pytest.raises(ValueError):
            MixingSumLayer(num_input_units=3, num_output_units=4)

    def test_initializes_parameters_with_expected_shape_and_normalization(self):
        """Init creates (F, H, K) weights normalized across H."""
        layer = MixingSumLayer(num_input_units=5, num_output_units=5, arity=3, num_folds=2)

        assert layer.params.shape == (2, 3, 5)
        # Normalization keeps mixture interpretation valid after initialization.
        assert torch.allclose(layer.params.sum(dim=1), torch.ones(2, 5), atol=1e-6)


class TestMixingSumLayerParameters:
    """Tests for MixingSumLayer parameter operations."""

    def test_reset_parameters_renormalizes_across_arity(self):
        """Reset keeps positive normalized weights across children."""
        layer = MixingSumLayer(num_input_units=4, num_output_units=4, arity=4, num_folds=3)

        with torch.no_grad():
            layer._params.fill_(1.0)

        layer.reset_parameters()

        assert torch.all(layer.params > 0)
        assert torch.allclose(layer.params.sum(dim=1), torch.ones(3, 4), atol=1e-6)

    def test_params_property_returns_internal_parameter(self):
        """Property should expose the exact backing tensor."""
        layer = MixingSumLayer(num_input_units=2, num_output_units=2, arity=2, num_folds=1)

        assert layer.params is layer._params


class TestMixingSumLayerForward:
    """Tests for MixingSumLayer forward pass."""

    def test_forward_matches_eval_mixing_without_mask(self):
        """Forward delegates correctly to eval_mixing."""
        layer = MixingSumLayer(num_input_units=3, num_output_units=3, arity=2, num_folds=2)
        x = torch.randn(2, 2, 3, 7)

        out = layer(x)
        expected = eval_mixing(x, layer.params, fold_mask=None)

        assert out.shape == (2, 3, 7)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_forward_with_fold_mask_matches_explicit_masked_sum(self):
        """Masked children contribute zero mass in probability space."""
        fold_mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32)
        layer = MixingSumLayer(
            num_input_units=2,
            num_output_units=2,
            arity=3,
            num_folds=2,
            fold_mask=fold_mask,
        )

        x = torch.randn(2, 3, 2, 5)
        out = layer(x)

        mask = fold_mask.view(2, 3, 1, 1)
        # Build an explicit probability-space baseline to verify masked children contribute nothing.
        explicit_prob = torch.einsum("fhk,fhkb->fkb", layer.params, torch.exp(x) * mask)
        expected = torch.log(explicit_prob)

        assert out.shape == (2, 2, 5)
        assert torch.allclose(out, expected, atol=1e-6)
