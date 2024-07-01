from itertools import product

from tests.fixtures import auto_set_test_seed
import unittest

from tests.fixtures import auto_set_test_seed
from spflow.meta.dispatch import init_default_sampling_context
from tests.utils.leaves import evaluate_log_likelihood, evaluate_samples
import pytest
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, marginalize
from spflow.meta.data import Scope
from spflow.modules.leaf import Gamma

# Constants
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> torch.Tensor:
    return torch.rand(out_features, out_channels), torch.rand(out_features, out_channels)


def make_module(alpha, beta) -> Gamma:
    scope = Scope(list(range(alpha.shape[0])))
    return Gamma(scope=scope, alpha=alpha, beta=beta)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_alpha(out_features: int, out_channels: int):
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(alpha=torch.full_like(alpha, -1.0), beta=beta)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_beta(out_features: int, out_channels: int):
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(alpha=alpha, beta=torch.full_like(beta, -1.0))


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_alpha(out_features: int, out_channels: int):
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(alpha=torch.full_like(alpha, 0.0), beta=beta)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_beta(out_features: int, out_channels: int):
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(alpha=alpha, beta=torch.full_like(beta, 0.0))


if __name__ == "__main__":
    unittest.main()
