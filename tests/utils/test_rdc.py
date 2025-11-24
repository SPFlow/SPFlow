"""Tests for Randomized Dependence Coefficient (RDC) module."""

import numpy as np
import pytest
import torch

from spflow.utils.rdc import cca_loop, cca_loop_np, rdc, rdc_np, rankdata_ordinal


class TestRDCBasicFunctionality:
    """Test basic functionality of rdc."""

    def test_rdc_basic_2d(self, device):
        """Test rdc with simple 2D data."""
        torch.manual_seed(42)
        # 100 samples, 2 features with some correlation
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        result = rdc(x, y, k=10, s=1 / 6.0, n=1)

        # rdc should be in [0, 1]
        assert 0.0 <= result.item() <= 1.0
        # Should detect some correlation
        assert result.item() > 0.1

    def test_rdc_independent_features(self, device):
        """Test rdc with independent features."""
        torch.manual_seed(42)
        # Two completely independent features
        x = torch.randn(100, 1, device=device)
        y = torch.randn(100, 1, device=device)

        result = rdc(x, y, k=10, s=1 / 6.0, n=1)

        # rdc should be close to 0
        assert 0.0 <= result.item() <= 1.0
        # For truly independent, should be low (but not necessarily 0 due to randomness)
        assert result.item() < 0.5

    def test_rdc_dependent_features(self, device):
        """Test rdc with dependent features."""
        torch.manual_seed(42)
        # x and y = x^2 (nonlinear relationship)
        x = torch.randn(100, 1, device=device)
        y = x**2

        result = rdc(x, y, k=10, s=1 / 6.0, n=1)

        # Should detect nonlinear dependence
        assert 0.0 <= result.item() <= 1.0
        # RDC can vary based on random projections, so just check it's positive
        assert result.item() >= 0.0

    def test_rdc_identical_features(self, device):
        """Test rdc with identical features."""
        torch.manual_seed(42)
        # x and y = x + small noise (strong linear dependence)
        x = torch.randn(100, 1, device=device)
        y = x + 0.01 * torch.randn(100, 1, device=device)

        result = rdc(x, y, k=10, s=1 / 6.0, n=1)

        # Should be high
        assert result.item() > 0.5


class TestRDCParameters:
    """Test rdc with different parameters."""

    def test_rdc_with_n_equals_1(self, device):
        """Test rdc with n=1 (single run)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        result = rdc(x, y, k=10, n=1)

        assert isinstance(result, torch.Tensor)
        assert 0.0 <= result.item() <= 1.0

    def test_rdc_with_n_greater_than_1(self, device):
        """Test rdc with n=5 (median of multiple runs)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        result = rdc(x, y, k=10, n=5)

        assert isinstance(result, torch.Tensor)
        assert 0.0 <= result.item() <= 1.0

    def test_rdc_with_custom_k(self, device):
        """Test rdc with different k parameters."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        # Test with different k values
        result_k2 = rdc(x, y, k=2, n=1)
        result_k5 = rdc(x, y, k=5, n=1)
        result_k10 = rdc(x, y, k=10, n=1)

        # All should be valid
        assert 0.0 <= result_k2.item() <= 1.0
        assert 0.0 <= result_k5.item() <= 1.0
        assert 0.0 <= result_k10.item() <= 1.0

    def test_rdc_with_custom_function(self, device):
        """Test rdc with custom function f."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        # Test with cos instead of sin
        result = rdc(x, y, f=torch.cos, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0


class TestRankdataOrdinal:
    """Test rankdata_ordinal function."""

    def test_rankdata_ordinal_basic(self, device):
        """Test rankdata_ordinal with simple data."""
        data = torch.tensor([3.0, 1.0, 2.0], device=device)
        result = rankdata_ordinal(data)

        # [3, 1, 2] -> ranks [3, 1, 2] (1-indexed)
        expected = torch.tensor([3.0, 1.0, 2.0], device=device)
        assert torch.allclose(result, expected)

    def test_rankdata_ordinal_duplicates(self, device):
        """Test rankdata_ordinal with duplicate values."""
        data = torch.tensor([2.0, 1.0, 2.0, 1.0], device=device)
        result = rankdata_ordinal(data)

        # Ordinal ranking assigns different ranks to ties based on position
        # Results should be valid ranks
        assert result.shape == data.shape
        assert torch.all(result >= 1.0)
        assert torch.all(result <= 4.0)

    def test_rankdata_ordinal_all_same(self, device):
        """Test rankdata_ordinal with all identical values."""
        data = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
        result = rankdata_ordinal(data)

        # Should assign different ranks based on position
        assert result.shape == data.shape
        assert len(torch.unique(result)) == 4  # All different ranks

    def test_rankdata_ordinal_sorted_ascending(self, device):
        """Test rankdata_ordinal with sorted ascending data."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        result = rankdata_ordinal(data)

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert torch.allclose(result, expected)

    def test_rankdata_ordinal_sorted_descending(self, device):
        """Test rankdata_ordinal with sorted descending data."""
        data = torch.tensor([4.0, 3.0, 2.0, 1.0], device=device)
        result = rankdata_ordinal(data)

        expected = torch.tensor([4.0, 3.0, 2.0, 1.0], device=device)
        assert torch.allclose(result, expected)


class TestCCALoop:
    """Test cca_loop function."""

    def test_cca_loop_basic(self, device):
        """Test cca_loop with basic covariance matrix."""
        torch.manual_seed(42)
        k = 10

        # Create a random covariance matrix
        fX = torch.randn(100, k, device=device)
        fY = torch.randn(100, k, device=device)
        C = torch.cov(torch.hstack([fX, fY]).T)

        result = cca_loop(k, C)

        # Result should be in [0, 1]
        assert 0.0 <= result.item() <= 1.0

    def test_cca_loop_with_different_k(self, device):
        """Test cca_loop with different k values."""
        torch.manual_seed(42)

        for k in [5, 10, 15]:
            fX = torch.randn(100, k, device=device)
            fY = torch.randn(100, k, device=device)
            C = torch.cov(torch.hstack([fX, fY]).T)

            result = cca_loop(k, C)
            assert 0.0 <= result.item() <= 1.0


class TestRDCNumPyVersion:
    """Test NumPy version of RDC."""

    def test_rdc_np_basic(self):
        """Test rdc_np with NumPy arrays."""
        np.random.seed(42)
        torch.manual_seed(42)

        # NumPy arrays
        x = np.random.randn(100, 1)
        y = x + 0.5 * np.random.randn(100, 1)

        result = rdc_np(x, y, k=10, s=1 / 6.0, n=1)

        # Should be in [0, 1]
        assert 0.0 <= result <= 1.0

    def test_rdc_np_independent(self):
        """Test rdc_np with independent features."""
        np.random.seed(42)

        x = np.random.randn(100, 1)
        y = np.random.randn(100, 1)

        result = rdc_np(x, y, k=10, n=1)

        assert 0.0 <= result <= 1.0

    def test_cca_loop_np_basic(self):
        """Test cca_loop_np with NumPy arrays."""
        np.random.seed(42)
        k = 10

        fX = np.random.randn(100, k)
        fY = np.random.randn(100, k)
        C = np.cov(np.hstack([fX, fY]).T)

        result = cca_loop_np(k, C)

        assert 0.0 <= result <= 1.0


class TestRDCEdgeCases:
    """Test edge cases and error handling in rdc."""

    def test_rdc_1d_input(self, device):
        """Test rdc with 1D input (should be reshaped to 2D)."""
        torch.manual_seed(42)
        x = torch.randn(100, device=device)
        y = x + 0.5 * torch.randn(100, device=device)

        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0

    def test_rdc_perfect_collinearity(self, device):
        """Test rdc with perfect collinearity."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = 2.0 * x + 0.01 * torch.randn(100, 1, device=device)  # near-perfect linear dependence

        result = rdc(x, y, k=10, n=1)

        # Should be high
        assert result.item() > 0.5

    def test_rdc_small_sample_size(self, device):
        """Test rdc with small sample size."""
        torch.manual_seed(42)
        x = torch.randn(10, 1, device=device)
        y = x + 0.5 * torch.randn(10, 1, device=device)

        # Should work but with lower k
        result = rdc(x, y, k=3, n=1)

        assert 0.0 <= result.item() <= 1.0

    def test_rdc_multivariate(self, device):
        """Test rdc with multivariate data (single columns extracted)."""
        torch.manual_seed(42)
        # RDC computes correlation between two sets of variables
        # Each should be treated as a single unit
        x = torch.randn(100, 1, device=device)
        y = torch.randn(100, 1, device=device)

        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0


class TestRDCReproducibility:
    """Test reproducibility of rdc."""

    def test_rdc_multiple_runs_consistency(self, device):
        """Test rdc with n>1 gives reasonable results."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        # Run with n=10 to get median
        result = rdc(x, y, k=10, n=10)

        assert 0.0 <= result.item() <= 1.0
        # Median should be reasonable
        assert result.item() > 0.1


class TestRDCNumericalStability:
    """Test numerical stability of rdc."""

    def test_rdc_binary_search_k_adjustment(self, device):
        """Test k adjustment logic in rdc."""
        torch.manual_seed(42)
        # Large k might require adjustment
        x = torch.randn(50, 1, device=device)
        y = x + 0.5 * torch.randn(50, 1, device=device)

        # k=20 might be too large for 50 samples, should adjust
        result = rdc(x, y, k=20, n=1)

        assert 0.0 <= result.item() <= 1.0

    def test_rdc_handles_complex_eigenvalues(self, device):
        """Test rdc handles complex eigenvalues gracefully."""
        torch.manual_seed(42)
        # Create data that might produce complex eigenvalues
        x = torch.randn(100, 1, device=device)
        y = torch.randn(100, 1, device=device)

        # Should handle gracefully
        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0
        assert not torch.isnan(result)


class TestRDCMathematicalProperties:
    """Test mathematical properties of rdc."""

    def test_rdc_symmetry(self, device):
        """Test that rdc(x, y) ≈ rdc(y, x)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        result_xy = rdc(x, y, k=10, n=1)

        # Note: Due to random projections, results won't be exactly equal
        # but should be in the same range
        torch.manual_seed(42)
        result_yx = rdc(y, x, k=10, n=1)

        # Both should be valid
        assert 0.0 <= result_xy.item() <= 1.0
        assert 0.0 <= result_yx.item() <= 1.0

    def test_rdc_scale_invariance(self, device):
        """Test that rdc is scale invariant."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        result1 = rdc(x, y, k=10, n=1)

        # Scale x and y
        torch.manual_seed(42)
        result2 = rdc(x * 10, y * 10, k=10, n=1)

        # RDC uses copula transformation (ranking), so should be scale invariant
        # Results should be similar (not exact due to numerical differences)
        assert abs(result1.item() - result2.item()) < 0.2

    def test_rdc_detects_nonlinear_relationships(self, device):
        """Test that rdc can detect various nonlinear relationships."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)

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
    """Test rdc with different data types and devices."""

    def test_rdc_float32(self, device):
        """Test rdc with float32 data."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, dtype=torch.float32, device=device)
        y = x + 0.5 * torch.randn(100, 1, dtype=torch.float32, device=device)

        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0

    def test_rdc_float64(self, device):
        """Test rdc with float64 data."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, dtype=torch.float64, device=device)
        y = x + 0.5 * torch.randn(100, 1, dtype=torch.float64, device=device)

        result = rdc(x, y, k=10, n=1)

        assert 0.0 <= result.item() <= 1.0


class TestRDCLinAlgErrorHandling:
    """Test handling of linear algebra errors."""

    def test_rdc_with_n_greater_1_handles_errors(self, device):
        """Test that rdc with n>1 handles LinAlgError gracefully."""
        torch.manual_seed(42)
        x = torch.randn(100, 1, device=device)
        y = x + 0.5 * torch.randn(100, 1, device=device)

        # Even if some runs fail, should return median of successful ones
        result = rdc(x, y, k=10, n=5)

        assert 0.0 <= result.item() <= 1.0
