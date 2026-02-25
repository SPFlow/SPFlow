"""Tests for tensorized layer utilities."""

import pytest
import torch

from spflow.zoo.pic.tensorized.utils import log_func_exp


class TestLogFuncExp:
    """Tests for log_func_exp function."""

    def test_single_tensor_sum_matches_logsumexp(self):
        """Test that log_func_exp with sum matches torch.logsumexp."""
        x = torch.randn(5, 10)

        result = log_func_exp(x, func=lambda t: t.sum(dim=1), dim=1, keepdim=False)
        expected = torch.logsumexp(x, dim=1)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_single_tensor_sum_keepdim(self):
        """Test keepdim=True preserves dimension."""
        x = torch.randn(3, 4, 5)

        result = log_func_exp(x, func=lambda t: t.sum(dim=1, keepdim=True), dim=1, keepdim=True)
        expected = torch.logsumexp(x, dim=1, keepdim=True)

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-5)

    def test_two_tensors_product_sum(self):
        """Test with two tensors (product then sum)."""
        left = torch.randn(3, 5)
        right = torch.randn(3, 5)

        def linear_func(l, r):
            return (l * r).sum(dim=1)

        result = log_func_exp(left, right, func=linear_func, dim=1, keepdim=False)

        # This identity is the contract that lets callers stay in log-space safely.
        expected = torch.logsumexp(left + right, dim=1)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_numerical_stability_large_values(self):
        """Test numerical stability with large values."""
        x = torch.tensor([1000.0, 1001.0, 1002.0])

        result = log_func_exp(x, func=lambda t: t.sum(dim=0), dim=0, keepdim=False)
        expected = torch.logsumexp(x, dim=0)

        assert torch.isfinite(result)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_numerical_stability_small_values(self):
        """Test numerical stability with very small (negative) values."""
        x = torch.tensor([-1000.0, -1001.0, -1002.0])

        result = log_func_exp(x, func=lambda t: t.sum(dim=0), dim=0, keepdim=False)
        expected = torch.logsumexp(x, dim=0)

        assert torch.isfinite(result)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_dimensions(self):
        """Test with batch dimensions."""
        x = torch.randn(2, 3, 4, 5)  # (B1, B2, F, K)

        result = log_func_exp(x, func=lambda t: t.sum(dim=-1), dim=-1, keepdim=False)
        expected = torch.logsumexp(x, dim=-1)

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-5)

    def test_gradients_flow(self):
        """Test that gradients flow correctly."""
        x = torch.randn(3, 4, requires_grad=True)

        result = log_func_exp(x, func=lambda t: t.sum(dim=1), dim=1, keepdim=False)
        loss = result.sum()
        loss.backward()

        # Stable gradients are required because tensorized layers are trained end-to-end.
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_empty_input_raises(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError):
            log_func_exp(func=lambda t: t, dim=0, keepdim=False)

    def test_einsum_style_function(self):
        """Test with einsum-style function (Tucker-like)."""
        left = torch.randn(2, 3, 4)  # (F, I, B)
        right = torch.randn(2, 3, 4)  # (F, J, B) where J=I
        weights = torch.rand(2, 3, 3, 5)  # (F, I, J, O)

        def tucker_linear(l, r):
            return torch.einsum("fib,fjb,fijo->fob", l, r, weights)

        result = log_func_exp(left, right, func=tucker_linear, dim=1, keepdim=True)

        assert result.shape == (2, 5, 4)  # (F, O, B)
        assert torch.isfinite(result).all()
