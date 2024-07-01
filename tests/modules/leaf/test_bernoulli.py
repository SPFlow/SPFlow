import unittest
from itertools import product

from spflow.exceptions import InvalidParameterCombinationError
from tests.fixtures import auto_set_test_seed

import pytest
import torch

from spflow import maximum_likelihood_estimation, marginalize
from spflow.meta.data import Scope
from spflow.modules.leaf.bernoulli import Bernoulli

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_module(p) -> Bernoulli:
    """
    Create a Bernoulli leaf node.

    Args:
        p: Probability of the distribution.
    """
    scope = Scope(list(range(p.shape[0])))
    return Bernoulli(scope=scope, p=p)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_p_greater_than_one(out_features: int, out_channels: int):
    """Test the constructor of a Bernoulli distribution with p greater than 1.0."""
    p = torch.rand(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(p=1.0 + p)


if __name__ == "__main__":
    unittest.main()
