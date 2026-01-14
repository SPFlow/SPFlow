"""Tests for functional sharing utilities."""

import torch

from spflow.exp.pic.functional_sharing import (
    FourierFeatures,
    SharedMLP,
    MultiHeadedMLP,
    FunctionGroup,
)


class TestFourierFeatures:
    """Tests for FourierFeatures layer."""

    def test_output_shape(self):
        """Test output has correct shape (2x input due to sin+cos)."""
        ff = FourierFeatures(in_features=3, out_features=16)
        x = torch.randn(10, 3)

        out = ff(x)

        assert out.shape == (10, 32)  # 16 * 2 for sin/cos

    def test_deterministic_with_same_input(self):
        """Test that same input produces same output."""
        ff = FourierFeatures(in_features=2, out_features=8)
        x = torch.randn(5, 2)

        out1 = ff(x)
        out2 = ff(x)

        assert torch.allclose(out1, out2)


class TestSharedMLP:
    """Tests for SharedMLP."""

    def test_output_shape(self):
        """Test output has hidden_dim shape."""
        mlp = SharedMLP(input_dim=2, hidden_dim=32, num_layers=2)
        x = torch.randn(10, 2)

        out = mlp(x)

        assert out.shape == (10, 32)

    def test_forward_no_nan(self):
        """Test forward pass produces no NaN values."""
        mlp = SharedMLP(input_dim=3, hidden_dim=16, num_layers=3)
        x = torch.randn(20, 3)

        out = mlp(x)

        assert not torch.isnan(out).any()


class TestMultiHeadedMLP:
    """Tests for MultiHeadedMLP."""

    def test_all_heads_output_shape(self):
        """Test all-heads output has correct shape."""
        shared = SharedMLP(input_dim=2, hidden_dim=16)
        multi = MultiHeadedMLP(shared, num_heads=5)
        x = torch.randn(10, 2)

        out = multi(x)  # All heads

        assert out.shape == (10, 5)

    def test_single_head_output_shape(self):
        """Test single-head output has correct shape."""
        shared = SharedMLP(input_dim=2, hidden_dim=16)
        multi = MultiHeadedMLP(shared, num_heads=5)
        x = torch.randn(10, 2)

        out = multi(x, head_idx=2)

        assert out.shape == (10, 1)

    def test_positive_outputs(self):
        """Test outputs are positive (softplus activation)."""
        shared = SharedMLP(input_dim=2, hidden_dim=16)
        multi = MultiHeadedMLP(shared, num_heads=3)
        x = torch.randn(10, 2)

        out = multi(x)

        assert (out > 0).all()


class TestFunctionGroup:
    """Tests for FunctionGroup."""

    def test_add_units(self):
        """Test adding units to group."""
        group = FunctionGroup(sharing_type="c")

        idx1 = group.add_unit("unit1")
        idx2 = group.add_unit("unit2")

        assert idx1 == 0
        assert idx2 == 1
        assert len(group.units) == 2

    def test_get_function_c_sharing(self):
        """Test getting C-shared function."""
        group = FunctionGroup(sharing_type="c", input_dim=2, hidden_dim=8)
        group.add_unit("unit1")
        group.add_unit("unit2")
        group.finalize()

        func = group.get_function(0)

        # Should return a callable
        assert callable(func)

        # Should work with tensor inputs
        z = torch.randn(3, 3, 1)
        y = torch.randn(3, 3, 1)
        out = func(z, y)

        assert out.shape == (3, 3)

    def test_get_function_f_sharing(self):
        """Test getting F-shared function."""
        group = FunctionGroup(sharing_type="f", input_dim=2, hidden_dim=8)
        group.add_unit("unit1")

        func = group.get_function(0)

        assert callable(func)

    def test_evaluate_batched_matches_per_head(self):
        """Test that evaluate_batched matches per-head get_function for C-sharing."""
        group = FunctionGroup(sharing_type="c", input_dim=2, hidden_dim=8)
        idx0 = group.add_unit("unit1")
        idx1 = group.add_unit("unit2")
        group.finalize()

        z = torch.randn(4, 5, 1)
        y = torch.randn(4, 5, 1)

        batched = group.evaluate_batched(z, y)
        assert batched.shape == (2, 4, 5)

        f0 = group.get_function(idx0)
        f1 = group.get_function(idx1)

        assert torch.allclose(batched[0], f0(z, y))
        assert torch.allclose(batched[1], f1(z, y))
