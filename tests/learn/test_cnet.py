"""Tests for CNet (Cutset Network) structure learning."""

import pytest
import torch
from torch import Tensor

from spflow.learn.cnet import learn_cnet
from spflow.modules.module import Module


def make_binary_data(n_samples: int = 200, n_vars: int = 3, seed: int = 42) -> Tensor:
    """Generate random binary data for testing."""
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n_samples, n_vars), generator=rng).float()


def make_categorical_data(
    n_samples: int = 200, n_vars: int = 3, cardinalities: list[int] | None = None, seed: int = 42
) -> tuple[Tensor, list[int]]:
    """Generate random categorical data for testing."""
    rng = torch.Generator().manual_seed(seed)
    if cardinalities is None:
        cardinalities = [3] * n_vars
    cols = []
    for K in cardinalities:
        cols.append(torch.randint(0, K, (n_samples,), generator=rng))
    return torch.stack(cols, dim=1).float(), cardinalities


class TestLearnCnetBasic:
    """Basic tests for learn_cnet function."""

    def test_learn_cnet_returns_valid_module(self):
        """Test that learn_cnet returns a valid Module."""
        data = make_binary_data(n_samples=100, n_vars=4)
        model = learn_cnet(data, cardinalities=2)

        assert isinstance(model, Module)
        assert tuple(model.scope.query) == (0, 1, 2, 3)

    def test_learn_cnet_log_likelihood_finite(self):
        """Test that log_likelihood returns finite values."""
        data = make_binary_data(n_samples=100, n_vars=4)
        model = learn_cnet(data, cardinalities=2, min_instances_slice=30)

        lls = model.log_likelihood(data[:8])
        assert lls.shape[0] == 8
        assert torch.isfinite(lls).all()

    def test_learn_cnet_variable_cardinalities(self):
        """Test CNet with mixed cardinalities."""
        data, cards = make_categorical_data(n_samples=100, n_vars=3, cardinalities=[2, 3, 4])
        model = learn_cnet(data, cardinalities=cards, min_instances_slice=30)

        assert isinstance(model, Module)
        lls = model.log_likelihood(data[:8])
        assert torch.isfinite(lls).all()


class TestCnetConditioningStrategies:
    """Test conditioning strategy support."""

    def test_cond_naive_mle_works(self):
        """Test that cond='naive_mle' works."""
        data = make_binary_data(n_samples=100, n_vars=4)
        model = learn_cnet(data, cardinalities=2, cond="naive_mle")
        assert isinstance(model, Module)

    def test_cond_random_works(self):
        """Test that cond='random' works."""
        data = make_binary_data(n_samples=100, n_vars=4)
        model = learn_cnet(data, cardinalities=2, cond="random", seed=42)
        assert isinstance(model, Module)

    def test_cond_invalid_raises(self):
        """Test that invalid cond raises ValueError."""
        data = make_binary_data(n_samples=100, n_vars=4)
        with pytest.raises(ValueError, match="Unknown conditioning strategy"):
            learn_cnet(data, cardinalities=2, cond="invalid")


class TestCnetMPE:
    """Test MPE (Most Probable Explanation) functionality."""

    def test_cnet_mpe_runs(self):
        """Test that MPE sampling runs without error."""
        data = make_binary_data(n_samples=100, n_vars=4)
        model = learn_cnet(data, cardinalities=2, min_instances_slice=30)

        # Create data with missing values
        test_data = data[:8].clone()
        test_data[0, 0] = float("nan")
        test_data[1, 1] = float("nan")
        test_data[2, :2] = float("nan")

        result = model.sample(data=test_data, is_mpe=True)

        # Check NaNs are filled
        assert not torch.isnan(result).any()
        # Check values are valid
        assert (result >= 0).all()
        assert (result < 2).all()

    def test_cnet_exact_mpe_matches_bruteforce_tiny(self):
        """Test MPE matches brute-force for tiny 2-variable binary problem."""
        # Create simple 2-variable binary data with clear pattern
        # Mostly (0,0) and (1,1), so MPE given one should predict the other
        torch.manual_seed(42)
        data = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],  # noise
            ]
        )

        model = learn_cnet(data, cardinalities=2, min_instances_slice=2, min_features_slice=1)

        # Test MPE: given x0=0, what is best x1?
        test_data = torch.tensor([[0.0, float("nan")]])
        result = model.sample(data=test_data, is_mpe=True)

        # Brute force: compute LL for all assignments
        lls = []
        for x1_val in [0, 1]:
            test = torch.tensor([[0.0, float(x1_val)]])
            ll = model.log_likelihood(test).sum()
            lls.append((x1_val, ll.item()))

        best_bruteforce = max(lls, key=lambda x: x[1])[0]

        # MPE should match brute-force
        assert result[0, 1].item() == best_bruteforce


class TestCnetSeedReproducibility:
    """Test seed reproducibility for random conditioning."""

    def test_seed_controls_random_conditioning(self):
        """Test that same seed produces identical results."""
        data = make_binary_data(n_samples=100, n_vars=5)

        model1 = learn_cnet(data, cardinalities=2, cond="random", seed=42)
        model2 = learn_cnet(data, cardinalities=2, cond="random", seed=42)

        # Same seed should produce same structure (same LL on test data)
        test_data = make_binary_data(n_samples=10, n_vars=5, seed=123)
        ll1 = model1.log_likelihood(test_data)
        ll2 = model2.log_likelihood(test_data)

        assert torch.allclose(ll1, ll2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds may produce different results."""
        data = make_binary_data(n_samples=100, n_vars=5)

        model1 = learn_cnet(data, cardinalities=2, cond="random", seed=42)
        model2 = learn_cnet(data, cardinalities=2, cond="random", seed=99)

        # Different seeds should likely produce different structures
        # (not guaranteed, but very likely with 5 variables)
        test_data = make_binary_data(n_samples=10, n_vars=5, seed=123)
        ll1 = model1.log_likelihood(test_data)
        ll2 = model2.log_likelihood(test_data)

        # This might rarely fail if the random choices happen to align
        # but with 5 variables it's extremely unlikely
        assert not torch.allclose(ll1, ll2)


class TestCnetInputValidation:
    """Test input validation."""

    def test_invalid_data_shape_raises(self):
        """Test that 1D data raises error."""
        data = torch.randn(100)
        with pytest.raises(Exception):  # InvalidParameterError
            learn_cnet(data, cardinalities=2)

    def test_nan_in_data_raises(self):
        """Test that NaN in training data raises error."""
        data = make_binary_data(n_samples=100, n_vars=4)
        data[0, 0] = float("nan")
        with pytest.raises(Exception):  # InvalidParameterError
            learn_cnet(data, cardinalities=2)

    def test_negative_values_raises(self):
        """Test that negative values raise error."""
        data = make_binary_data(n_samples=100, n_vars=4)
        data[0, 0] = -1
        with pytest.raises(Exception):  # InvalidParameterError
            learn_cnet(data, cardinalities=2)

    def test_value_exceeds_cardinality_raises(self):
        """Test that values >= cardinality raise error."""
        data = make_binary_data(n_samples=100, n_vars=4)
        data[0, 0] = 2  # Exceeds cardinality of 2
        with pytest.raises(Exception):  # InvalidParameterError
            learn_cnet(data, cardinalities=2)

    def test_cardinalities_length_mismatch_raises(self):
        """Test that cardinalities length mismatch raises error."""
        data = make_binary_data(n_samples=100, n_vars=4)
        with pytest.raises(Exception):  # InvalidParameterError
            learn_cnet(data, cardinalities=[2, 2])  # Only 2 instead of 4
