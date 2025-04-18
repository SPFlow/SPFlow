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
from spflow.modules.leaf import Categorical


out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_module(p) -> Categorical:
    """
    Create a Categorical leaf node.

    Args:
        p: Probability of the distribution.
    """
    scope = Scope(list(range(p.shape[0])))
    return Categorical(scope=scope, p=p)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_p_greater_than_one(out_features: int, out_channels: int, device):
    """Test the constructor of a Bernoulli distribution with p greater than 1.0."""
    p = torch.rand(out_features, out_channels).to(device)
    with pytest.raises(ValueError):
        make_module(p=1.5 + p).to(device)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_p_smaller_than_zero(out_features: int, out_channels: int, device):
    """Test the constructor of a Bernoulli distribution with p smaller than 1.0."""
    p = torch.rand(out_features, out_channels).to(device)
    with pytest.raises(ValueError):
        make_module(p=p - 1.5).to(device)
