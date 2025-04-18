from tests.fixtures import auto_set_test_seed
import unittest
from itertools import product

from typing import Union
from spflow.meta.dispatch import init_default_sampling_context
from tests.utils.leaves import evaluate_log_likelihood, evaluate_samples
import pytest
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, marginalize
from spflow.meta.data import Scope
from spflow.modules.leaf import Poisson

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int, device) -> torch.Tensor:
    return torch.rand(out_features, out_channels, device=device)


def make_module(rate) -> Poisson:
    scope = Scope(list(range(rate.shape[0])))
    return Poisson(scope=scope, rate=rate)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_rate(out_features: int, out_channels: int, device):
    rate = make_params(out_features, out_channels, device)
    with pytest.raises(ValueError):
        make_module(rate=torch.full_like(rate, -1.0)).to(device)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_rate(out_features: int, out_channels: int, device):
    rate = make_params(out_features, out_channels, device)
    with pytest.raises(ValueError):
        make_module(rate=torch.full_like(rate, 0.0)).to(device)
