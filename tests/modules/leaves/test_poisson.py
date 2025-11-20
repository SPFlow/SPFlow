from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.leaves import Poisson

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int, num_repetitions: int) -> torch.Tensor:
    return torch.rand(out_features, out_channels, num_repetitions)


def make_module(rate) -> Poisson:
    scope = Scope(list(range(rate.shape[0])))
    return Poisson(scope=scope, rate=rate)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_rate(out_features: int, out_channels: int, num_repetitions: int):
    rate = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_module(rate=torch.full_like(rate, -1.0)).distribution
