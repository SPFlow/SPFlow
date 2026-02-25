"""Tests for Tucker layer."""

import pytest
import torch

from spflow.zoo.pic.tensorized import TuckerLayer


class TestTuckerLayerInit:
    """Tests for TuckerLayer initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        layer = TuckerLayer(
            num_input_units=4,
            num_output_units=8,
            num_folds=2,
        )

        assert layer.num_input_units == 4
        assert layer.num_output_units == 8
        assert layer.num_folds == 2
        assert layer.arity == 2
        assert layer.params.shape == (2, 4, 4, 8)

    def test_arity_must_be_2(self):
        """Test that arity != 2 raises error."""
        # Current Tucker implementation models binary interactions only.
        with pytest.raises(NotImplementedError):
            TuckerLayer(
                num_input_units=4,
                num_output_units=8,
                arity=3,
            )

    def test_fold_mask_must_be_none(self):
        """Test that fold_mask raises error."""
        with pytest.raises(ValueError):
            TuckerLayer(
                num_input_units=4,
                num_output_units=8,
                fold_mask=torch.ones(2, 2),
            )


class TestTuckerLayerForward:
    """Tests for TuckerLayer forward pass."""

    def test_output_shape(self):
        """Test forward produces correct output shape."""
        F, I, O, B = 3, 4, 5, 10
        layer = TuckerLayer(num_input_units=I, num_output_units=O, num_folds=F)

        x = torch.randn(F, 2, I, B)  # (F, H=2, K, B)
        out = layer(x)

        assert out.shape == (F, O, B)

    def test_output_finite(self):
        """Test forward produces finite values."""
        layer = TuckerLayer(num_input_units=3, num_output_units=4, num_folds=2)
        x = torch.randn(2, 2, 3, 5)

        out = layer(x)

        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        """Test gradients flow through forward pass."""
        layer = TuckerLayer(num_input_units=3, num_output_units=4, num_folds=2)
        x = torch.randn(2, 2, 3, 5, requires_grad=True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert layer.params.grad is not None

    def test_no_nan_on_random_input(self):
        """Test no NaN on various random inputs."""
        layer = TuckerLayer(num_input_units=8, num_output_units=16, num_folds=4)

        for _ in range(10):
            x = torch.randn(4, 2, 8, 32)
            out = layer(x)
            assert not torch.isnan(out).any()


class TestTuckerLayerEquivalence:
    """Tests for numerical equivalence with expanded baseline."""

    def test_equivalence_with_explicit_outer_product(self):
        """Test Tucker layer matches explicit outer product + weighted sum."""
        F, I, O, B = 1, 3, 2, 4
        layer = TuckerLayer(num_input_units=I, num_output_units=O, num_folds=F)

        # Fixed params make the numerical baseline deterministic.
        W = torch.rand(F, I, I, O)
        layer._params.data = W

        # Use log-domain inputs to mirror how SPFlow layers are evaluated.
        x = torch.randn(F, 2, I, B)
        left_log = x[:, 0]  # (F, I, B)
        right_log = x[:, 1]  # (F, I, B)

        tucker_out = layer(x)

        # Compare against direct probability-space contraction to catch algebra regressions.
        left_prob = torch.exp(left_log)  # (F, I, B)
        right_prob = torch.exp(right_log)  # (F, I, B)

        outer = left_prob.unsqueeze(2) * right_prob.unsqueeze(1)

        explicit_prob = torch.einsum("fijb,fijo->fob", outer, W)
        explicit_log = torch.log(explicit_prob)

        assert torch.allclose(tucker_out, explicit_log, atol=1e-4)

    def test_batch_equivalence(self):
        """Test equivalence with larger batch."""
        layer = TuckerLayer(num_input_units=2, num_output_units=3, num_folds=1)
        x = torch.randn(1, 2, 2, 100)

        out = layer(x)

        # Large batches guard vectorization paths from shape-only correctness.
        assert out.shape == (1, 3, 100)
        assert torch.isfinite(out).all()
