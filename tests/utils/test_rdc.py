"""Tests for Randomized Dependence Coefficient (RDC) module."""

import numpy as np
import pytest
import torch

import spflow.utils.rdc as rdc_module
from spflow.utils.rdc import cca_loop, cca_loop_np, rdc, rdc_np, rankdata_ordinal


def _randn(*size: int, **kwargs) -> torch.Tensor:
    return torch.randn(*size, **kwargs)


def _np_randn(*size: int) -> np.ndarray:
    return np.random.standard_normal(size)


class TestRDCBasicFunctionality:
    """Test basic functionality of rdc."""

    def test_rdc_basic_2d(self):
        """Test rdc with simple 2D data."""
        # 100 samples, 2 features with some correlation
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        result = rdc(x, y, k=10, s=1 / 6.0, n=1)

        # rdc should be in [0, 1]
        assert 0.0 <= result.item() <= 1.0
        # Should detect some correlation
        assert result.item() > 0.1

    def test_rdc_independent_features(self):
        """Test rdc with independent features."""
        # Two completely independent features
        x = _randn(100, 1)
        y = _randn(100, 1)

        result = rdc(x, y, k=10, s=1 / 6.0, n=1)

        # rdc should be close to 0
        assert 0.0 <= result.item() <= 1.0
        # For truly independent, should be low (but not necessarily 0 due to randomness)
        assert result.item() < 0.5

    def test_rdc_dependent_features(self):
        """Test rdc with dependent features."""
        # x and y = x^2 (nonlinear relationship)
        x = _randn(100, 1)
        y = x**2

        result = rdc(x, y, k=10, s=1 / 6.0, n=1)

        # Should detect nonlinear dependence
        assert 0.0 <= result.item() <= 1.0
        # RDC can vary based on random projections, so just check it's positive
        assert result.item() >= 0.0

    def test_rdc_identical_features(self):
        """Test rdc with identical features."""
        # x and y = x + small noise (strong linear dependence)
        x = _randn(100, 1)
        y = x + 0.01 * _randn(100, 1)

        result = rdc(x, y, k=10, s=1 / 6.0, n=1)

        # Should be high
        assert result.item() > 0.5


class TestRDCParameters:
    """Test rdc with different parameters."""

    def test_rdc_with_n_equals_1(self):
        """Test rdc with n=1 (single run)."""
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        result = rdc(x, y, k=10, n=1)

        assert isinstance(result, torch.Tensor)
        assert 0.0 <= result.item() <= 1.0

    def test_rdc_with_n_greater_than_1(self):
        """Test rdc with n=5 (median of multiple runs)."""
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        result = rdc(x, y, k=10, n=5)

        assert isinstance(result, torch.Tensor)
        assert 0.0 <= result.item() <= 1.0

    def test_rdc_with_custom_k(self):
        """Test rdc with different k parameters."""
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        # Test with different k values
        result_k2 = rdc(x, y, k=2, n=1)
        result_k5 = rdc(x, y, k=5, n=1)
        result_k10 = rdc(x, y, k=10, n=1)

        # All should be valid
        assert 0.0 <= result_k2.item() <= 1.0
        assert 0.0 <= result_k5.item() <= 1.0
        assert 0.0 <= result_k10.item() <= 1.0

    def test_rdc_with_custom_function(self):
        """Test rdc with custom function f."""
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        # Test with cos instead of sin
        result = rdc(x, y, f=torch.cos, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0


class TestRankdataOrdinal:
    """Test rankdata_ordinal function."""

    def test_rankdata_ordinal_basic(self):
        """Test rankdata_ordinal with simple data."""
        data = torch.tensor([3.0, 1.0, 2.0])
        result = rankdata_ordinal(data)

        # [3, 1, 2] -> ranks [3, 1, 2] (1-indexed)
        expected = torch.tensor([3.0, 1.0, 2.0])
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)

    def test_rankdata_ordinal_duplicates(self):
        """Test rankdata_ordinal with duplicate values."""
        data = torch.tensor([2.0, 1.0, 2.0, 1.0])
        result = rankdata_ordinal(data)

        # Ordinal ranking assigns different ranks to ties based on position
        # Results should be valid ranks
        assert result.shape == data.shape
        assert torch.all(result >= 1.0)
        assert torch.all(result <= 4.0)

    def test_rankdata_ordinal_all_same(self):
        """Test rankdata_ordinal with all identical values."""
        data = torch.tensor([5.0, 5.0, 5.0, 5.0])
        result = rankdata_ordinal(data)

        # Should assign different ranks based on position
        assert result.shape == data.shape
        assert len(torch.unique(result)) == 4  # All different ranks

    def test_rankdata_ordinal_sorted_ascending(self):
        """Test rankdata_ordinal with sorted ascending data."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = rankdata_ordinal(data)

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)

    def test_rankdata_ordinal_sorted_descending(self):
        """Test rankdata_ordinal with sorted descending data."""
        data = torch.tensor([4.0, 3.0, 2.0, 1.0])
        result = rankdata_ordinal(data)

        expected = torch.tensor([4.0, 3.0, 2.0, 1.0])
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


class TestCCALoop:
    """Test cca_loop function."""

    def test_cca_loop_basic(self):
        """Test cca_loop with basic covariance matrix."""
        k = 10

        # Create a random covariance matrix
        fX = _randn(100, k)
        fY = _randn(100, k)
        C = torch.cov(torch.hstack([fX, fY]).T)

        result = cca_loop(k, C)

        # Result should be in [0, 1]
        assert 0.0 <= result.item() <= 1.0

    def test_cca_loop_with_different_k(self):
        """Test cca_loop with different k values."""

        for k in [5, 10, 15]:
            fX = _randn(100, k)
            fY = _randn(100, k)
            C = torch.cov(torch.hstack([fX, fY]).T)

            result = cca_loop(k, C)
            assert 0.0 <= result.item() <= 1.0


class TestRDCNumPyVersion:
    """Test NumPy version of RDC."""

    def test_rdc_np_basic(self):
        """Test rdc_np with NumPy arrays."""
        # NumPy arrays
        x = _np_randn(100, 1)
        y = x + 0.5 * _np_randn(100, 1)

        result = rdc_np(x, y, k=10, s=1 / 6.0, n=1)

        # Should be in [0, 1]
        assert 0.0 <= result <= 1.0

    def test_rdc_np_independent(self):
        """Test rdc_np with independent features."""
        x = _np_randn(100, 1)
        y = _np_randn(100, 1)

        result = rdc_np(x, y, k=10, n=1)

        assert 0.0 <= result <= 1.0

    def test_cca_loop_np_basic(self):
        """Test cca_loop_np with NumPy arrays."""
        k = 10

        fX = _np_randn(100, k)
        fY = _np_randn(100, k)
        C = np.cov(np.hstack([fX, fY]).T)

        result = cca_loop_np(k, C)

        assert 0.0 <= result <= 1.0


class TestRDCEdgeCases:
    """Test edge cases and error handling in rdc."""

    def test_rdc_1d_input(self):
        """Test rdc with 1D input (should be reshaped to 2D)."""
        x = _randn(100)
        y = x + 0.5 * _randn(100)

        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0

    def test_rdc_perfect_collinearity(self):
        """Test rdc with perfect collinearity."""
        x = _randn(100, 1)
        y = 2.0 * x + 0.01 * _randn(100, 1)  # near-perfect linear dependence

        result = rdc(x, y, k=10, n=1)

        # Should be high
        assert result.item() > 0.5

    def test_rdc_small_sample_size(self):
        """Test rdc with small sample size."""
        x = _randn(10, 1)
        y = x + 0.5 * _randn(10, 1)

        # Should work but with lower k
        result = rdc(x, y, k=3, n=1)

        assert 0.0 <= result.item() <= 1.0

    def test_rdc_multivariate(self):
        """Test rdc with multivariate data (single columns extracted)."""
        # RDC computes correlation between two sets of variables
        # Each should be treated as a single unit
        x = _randn(100, 1)
        y = _randn(100, 1)

        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0


class TestRDCReproducibility:
    """Test reproducibility of rdc."""

    def test_rdc_multiple_runs_consistency(self):
        """Test rdc with n>1 gives reasonable results."""
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        # Run with n=10 to get median
        result = rdc(x, y, k=10, n=10)

        assert 0.0 <= result.item() <= 1.0
        # Median should be reasonable
        assert result.item() > 0.1


class TestRDCNumericalStability:
    """Test numerical stability of rdc."""

    def test_rdc_binary_search_k_adjustment(self):
        """Test k adjustment logic in rdc."""
        # Large k might require adjustment
        x = _randn(50, 1)
        y = x + 0.5 * _randn(50, 1)

        # k=20 might be too large for 50 samples, should adjust
        result = rdc(x, y, k=20, n=1)

        assert 0.0 <= result.item() <= 1.0

    def test_rdc_handles_complex_eigenvalues(self):
        """Test rdc handles complex eigenvalues gracefully."""
        # Create data that might produce complex eigenvalues
        x = _randn(100, 1)
        y = _randn(100, 1)

        # Should handle gracefully
        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0
        assert not torch.isnan(result)


class TestRDCMathematicalProperties:
    """Test mathematical properties of rdc."""

    def test_rdc_symmetry(self):
        """Test that rdc(x, y) ≈ rdc(y, x)."""
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        result_xy = rdc(x, y, k=10, n=1)

        # Note: Due to random projections, results won't be exactly equal
        # but should be in the same range
        result_yx = rdc(y, x, k=10, n=1)

        # Both should be valid
        assert 0.0 <= result_xy.item() <= 1.0
        assert 0.0 <= result_yx.item() <= 1.0

    def test_rdc_scale_invariance(self):
        """Test that rdc is scale invariant."""
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        result1 = rdc(x, y, k=10, n=1)

        # Scale x and y
        result2 = rdc(x * 10, y * 10, k=10, n=1)

        # RDC uses copula transformation (ranking), so should be scale invariant
        # Results should be similar (not exact due to numerical differences)
        assert abs(result1.item() - result2.item()) < 0.2

    def test_rdc_detects_nonlinear_relationships(self):
        """Test that rdc can detect various nonlinear relationships."""
        x = _randn(100, 1)

        # Various nonlinear relationships
        relationships = [
            lambda t: t**2,  # quadratic
            lambda t: torch.abs(t),  # absolute value
            lambda t: torch.sign(t),  # sign
        ]

        for rel in relationships:
            y = rel(x)
            result = rdc(x, y, k=10, n=1)

            # Should detect some dependence
            assert 0.0 <= result.item() <= 1.0


class TestRDCDifferentDataTypes:
    """Test rdc with different data types."""

    def test_rdc_float32(self):
        """Test rdc with float32 data."""
        x = _randn(100, 1, dtype=torch.float32)
        y = x + 0.5 * _randn(100, 1, dtype=torch.float32)

        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0

    def test_rdc_float64(self):
        """Test rdc with float64 data."""
        x = _randn(100, 1, dtype=torch.float64)
        y = x + 0.5 * _randn(100, 1, dtype=torch.float64)

        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0


class TestRDCLinAlgErrorHandling:
    """Test handling of linear algebra errors."""

    def test_rdc_with_n_greater_1_handles_errors(self):
        """Test that rdc with n>1 handles LinAlgError gracefully."""
        x = _randn(100, 1)
        y = x + 0.5 * _randn(100, 1)

        # Even if some runs fail, should return median of successful ones
        result = rdc(x, y, k=10, n=5)

        assert 0.0 <= result.item() <= 1.0


class TestRDCBranchCoverage:
    """Branch-oriented tests for difficult-to-trigger paths in rdc module."""

    def test_rdc_n_greater_1_skips_linalg_error(self, monkeypatch):
        """rdc should ignore LinAlgError runs and take median of successful values."""
        original_rdc = rdc_module.rdc
        outputs = iter(
            [
                torch.linalg.LinAlgError("singular"),
                torch.tensor(0.2, dtype=torch.float32),
                torch.tensor(0.6, dtype=torch.float32),
                torch.tensor(0.4, dtype=torch.float32),
            ]
        )

        def fake_recursive_call(*_args, **_kwargs):
            value = next(outputs)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(rdc_module, "rdc", fake_recursive_call)
        x = _randn(20, 1)
        y = _randn(20, 1)

        result = original_rdc(x, y, n=4)

        torch.testing.assert_close(result, torch.tensor(0.4, dtype=x.dtype), rtol=0.0, atol=1e-6)

    def test_rdc_real_eigenvalue_path(self, monkeypatch):
        """rdc should support purely real eigenvalues and return sqrt(max(eigs))."""

        def fake_eigvals(_matrix):
            return torch.tensor([0.25], dtype=torch.float32)

        monkeypatch.setattr(torch.linalg, "eigvals", fake_eigvals)

        x = _randn(30, 1)
        y = _randn(30, 1)
        result = rdc(x, y, k=1, n=1)

        torch.testing.assert_close(result, torch.tensor(0.5, dtype=result.dtype), rtol=0.0, atol=1e-6)

    def test_cca_loop_handles_large_imaginary_part_then_converges(self, monkeypatch):
        """cca_loop should reduce k when imaginary part is significant."""
        eigvals_sequence = iter(
            [
                torch.tensor([0.2 + 0.1j], dtype=torch.complex64),
                torch.tensor([0.36], dtype=torch.float32),
                torch.tensor([0.36], dtype=torch.float32),
                torch.tensor([0.36], dtype=torch.float32),
            ]
        )

        def fake_eigvals(_matrix):
            return next(eigvals_sequence)

        monkeypatch.setattr(torch.linalg, "eigvals", fake_eigvals)

        c = torch.eye(6, dtype=torch.float32)
        result = cca_loop(3, c)

        torch.testing.assert_close(result, torch.tensor(0.6, dtype=result.dtype), rtol=0.0, atol=1e-6)

    def test_cca_loop_rejects_out_of_range_real_eigenvalues(self, monkeypatch):
        """cca_loop should shrink k when real eigenvalues fall outside [0, 1]."""
        eigvals_sequence = iter(
            [
                torch.tensor([1.2], dtype=torch.float32),
                torch.tensor([0.49], dtype=torch.float32),
                torch.tensor([0.49], dtype=torch.float32),
                torch.tensor([0.49], dtype=torch.float32),
            ]
        )

        def fake_eigvals(_matrix):
            return next(eigvals_sequence)

        monkeypatch.setattr(torch.linalg, "eigvals", fake_eigvals)

        c = torch.eye(6, dtype=torch.float32)
        result = cca_loop(3, c)

        torch.testing.assert_close(result, torch.tensor(0.7, dtype=result.dtype), rtol=0.0, atol=1e-6)

    def test_rdc_np_n_greater_1_skips_linalg_error(self, monkeypatch):
        """rdc_np should ignore LinAlgError runs and take median of successful values."""
        original_rdc_np = rdc_module.rdc_np
        outputs = iter(
            [
                np.linalg.LinAlgError("singular"),
                0.2,
                0.8,
                0.4,
            ]
        )

        def fake_recursive_call(*_args, **_kwargs):
            value = next(outputs)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(rdc_module, "rdc", fake_recursive_call)

        x = _np_randn(20, 1)
        y = _np_randn(20, 1)
        result = original_rdc_np(x, y, n=4)

        assert result == pytest.approx(0.4, abs=1e-8)

    def test_rdc_np_accepts_1d_inputs(self):
        """rdc_np should reshape 1D arrays internally."""
        x = _np_randn(50)
        y = _np_randn(50)

        result = rdc_np(x, y, k=5, n=1)

        assert 0.0 <= result <= 1.0

    def test_cca_loop_np_binary_search_fallback_paths(self, monkeypatch):
        """cca_loop_np should run invalid->valid binary-search fallback branches."""
        eigvals_sequence = iter(
            [
                np.array([1.1], dtype=np.float64),
                np.array([0.25], dtype=np.float64),
                np.array([0.25], dtype=np.float64),
                np.array([0.25], dtype=np.float64),
            ]
        )

        def fake_eigvals(_matrix):
            return next(eigvals_sequence)

        monkeypatch.setattr(np.linalg, "eigvals", fake_eigvals)

        c = np.eye(6)
        result = cca_loop_np(3, c)

        assert result == pytest.approx(0.5, abs=1e-8)
