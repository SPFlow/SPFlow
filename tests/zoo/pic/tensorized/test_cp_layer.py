"""Tests for CP layer variants."""

import pytest
import torch

from spflow.zoo.pic.tensorized.cp import (
    CollapsedCPLayer,
    CPLayer,
    SharedCPLayer,
    UncollapsedCPLayer,
)


class TestCollapsedCPLayerInit:
    """Tests for CollapsedCPLayer initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        layer = CollapsedCPLayer(
            num_input_units=4,
            num_output_units=8,
            arity=2,
            num_folds=3,
        )

        assert layer.num_input_units == 4
        assert layer.num_output_units == 8
        assert layer.arity == 2
        assert layer.num_folds == 3
        assert layer.params.shape == (3, 2, 4, 8)

    def test_higher_arity(self):
        """Test with arity > 2."""
        layer = CollapsedCPLayer(
            num_input_units=3,
            num_output_units=5,
            arity=4,
            num_folds=2,
        )

        assert layer.arity == 4
        assert layer.params.shape == (2, 4, 3, 5)


class TestCollapsedCPLayerForward:
    """Tests for CollapsedCPLayer forward pass."""

    def test_output_shape(self):
        """Test forward produces correct output shape."""
        F, H, I, O, B = 3, 2, 4, 5, 10
        layer = CollapsedCPLayer(num_input_units=I, num_output_units=O, arity=H, num_folds=F)

        x = torch.randn(F, H, I, B)
        out = layer(x)

        assert out.shape == (F, O, B)

    def test_output_finite(self):
        """Test forward produces finite values."""
        layer = CollapsedCPLayer(num_input_units=3, num_output_units=4, num_folds=2)
        x = torch.randn(2, 2, 3, 5)

        out = layer(x)

        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        """Test gradients flow through forward pass."""
        layer = CollapsedCPLayer(num_input_units=3, num_output_units=4, num_folds=2)
        x = torch.randn(2, 2, 3, 5, requires_grad=True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_with_fold_mask(self):
        """Test with fold mask."""
        F, H = 2, 3
        fold_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
        layer = CollapsedCPLayer(
            num_input_units=4,
            num_output_units=5,
            arity=H,
            num_folds=F,
            fold_mask=fold_mask,
        )

        x = torch.randn(F, H, 4, 8)
        out = layer(x)

        assert out.shape == (F, 5, 8)
        assert torch.isfinite(out).all()


class TestUncollapsedCPLayer:
    """Tests for UncollapsedCPLayer."""

    def test_basic_init(self):
        """Test basic initialization."""
        layer = UncollapsedCPLayer(
            num_input_units=4,
            num_output_units=8,
            arity=2,
            num_folds=3,
            rank=5,
        )

        assert layer.rank == 5
        assert layer.params_in.shape == (3, 2, 4, 5)
        assert layer.params_out.shape == (3, 5, 8)

    def test_output_shape(self):
        """Test forward produces correct output shape."""
        layer = UncollapsedCPLayer(num_input_units=4, num_output_units=6, num_folds=2, rank=3)

        x = torch.randn(2, 2, 4, 10)
        out = layer(x)

        assert out.shape == (2, 6, 10)

    def test_rank_must_be_positive(self):
        """Test that rank <= 0 raises error."""
        # Rank-0 would erase all signal; constructor should reject this early.
        with pytest.raises(ValueError):
            UncollapsedCPLayer(
                num_input_units=4,
                num_output_units=8,
                rank=0,
            )


class TestSharedCPLayer:
    """Tests for SharedCPLayer."""

    def test_basic_init(self):
        """Test basic initialization."""
        layer = SharedCPLayer(
            num_input_units=4,
            num_output_units=8,
            arity=2,
            num_folds=3,
        )

        # Shared variant should not allocate per-fold parameters.
        assert layer.params.shape == (2, 4, 8)

    def test_output_shape(self):
        """Test forward produces correct output shape."""
        layer = SharedCPLayer(num_input_units=4, num_output_units=6, num_folds=3, arity=2)

        x = torch.randn(3, 2, 4, 10)
        out = layer(x)

        assert out.shape == (3, 6, 10)

    def test_fold_invariance(self):
        """Test that shared params produce consistent behavior across folds."""
        layer = SharedCPLayer(num_input_units=3, num_output_units=4, num_folds=4)

        # Repeating inputs isolates fold-specific effects from input variation.
        x_single = torch.randn(1, 2, 3, 5)
        x_repeated = x_single.repeat(4, 1, 1, 1)

        out = layer(x_repeated)

        # Any difference would indicate unintended fold-dependent behavior.
        for f in range(1, 4):
            assert torch.allclose(out[0], out[f], atol=1e-6)


class TestCPLayerFactory:
    """Tests for CPLayer factory function."""

    def test_collapsed_default(self):
        """Test default returns CollapsedCPLayer."""
        layer = CPLayer(num_input_units=4, num_output_units=8)
        assert isinstance(layer, CollapsedCPLayer)

    def test_uncollapsed(self):
        """Test collapsed=False returns UncollapsedCPLayer."""
        layer = CPLayer(num_input_units=4, num_output_units=8, collapsed=False, rank=3)
        assert isinstance(layer, UncollapsedCPLayer)
        assert layer.rank == 3

    def test_shared(self):
        """Test shared=True returns SharedCPLayer."""
        layer = CPLayer(num_input_units=4, num_output_units=8, shared=True)
        assert isinstance(layer, SharedCPLayer)

    def test_shared_uncollapsed_raises(self):
        """Test shared + uncollapsed raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            CPLayer(
                num_input_units=4,
                num_output_units=8,
                shared=True,
                collapsed=False,
            )


class TestCPLayerEquivalence:
    """Tests for numerical equivalence."""

    def test_collapsed_cp_explicit(self):
        """Test collapsed CP matches explicit per-child weighted sum."""
        F, H, I, O, B = 1, 2, 3, 4, 5
        layer = CollapsedCPLayer(num_input_units=I, num_output_units=O, arity=H, num_folds=F)

        x = torch.randn(F, H, I, B)

        cp_out = layer(x)

        # Baseline mirrors CP semantics in probability space for regression safety.
        W = layer.params  # (F, H, I, O)
        exp_x = torch.exp(x)  # (F, H, I, B)

        weighted = torch.einsum("fhio,fhib->fhob", W, exp_x)

        prod = weighted.prod(dim=1)
        explicit_out = torch.log(prod)

        assert torch.allclose(cp_out, explicit_out, atol=1e-4)
