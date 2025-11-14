from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.leaves.geometric import Geometric

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_module(p) -> Geometric:
    """Create a Categorical leaves node.

    Args:
        p: Probability of the distribution.
    """
    scope = Scope(list(range(p.shape[0])))
    return Geometric(scope=scope, p=p)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_p_greater_than_one(out_features: int, out_channels: int):
    """Test the constructor of a Bernoulli distribution with p greater than 1.0."""
    p = torch.rand(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(p=1.5 + p)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_p_smaller_than_zero(out_features: int, out_channels: int):
    """Test the constructor of a Bernoulli distribution with p smaller than 1.0."""
    p = torch.rand(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(p=p - 1.5)
