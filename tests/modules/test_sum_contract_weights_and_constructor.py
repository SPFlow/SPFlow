import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError, InvalidWeightsError, ShapeError
from spflow.modules.sums import Sum
from spflow.modules.sums.elementwise_sum import ElementwiseSum
from tests.modules.module_contract_data import SUM_PARAMS
from tests.modules.test_helpers.builders import (
    build_elementwise_sum,
    build_sum,
    normalized_sum_weights,
)
from tests.utils.leaves import make_normal_leaf

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", SUM_PARAMS)
def test_sum_weights_are_normalized_contract(
    in_channels: int, out_channels: int, out_features: int, num_reps: int
):
    weights = normalized_sum_weights(
        out_features=out_features,
        in_channels=in_channels,
        out_channels=out_channels,
        num_repetitions=num_reps,
    )
    module = Sum(
        inputs=make_normal_leaf(out_features=out_features, out_channels=in_channels), weights=weights
    )
    torch.testing.assert_close(
        module.weights.sum(dim=module.sum_dim), torch.ones_like(module.weights.sum(dim=module.sum_dim))
    )


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", SUM_PARAMS)
@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_constructor_rejects_invalid_weights(
    in_channels: int, out_channels: int, out_features: int, num_reps: int, builder
):
    module = builder(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    with pytest.raises((InvalidWeightsError, ShapeError, ValueError)):
        module.weights = torch.zeros_like(module.weights)


@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_constructor_rejects_empty_inputs(builder):
    module_cls = Sum if builder is build_sum else ElementwiseSum
    with pytest.raises(ValueError):
        module_cls(inputs=[], out_channels=2)


@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_constructor_rejects_invalid_out_channels(builder):
    module_cls = Sum if builder is build_sum else ElementwiseSum
    leaf = make_normal_leaf(out_features=1, out_channels=2)
    with pytest.raises(ValueError):
        module_cls(inputs=[leaf], out_channels=0)


def test_elementwise_sum_rejects_num_repetitions_with_weights():
    leaves = [
        make_normal_leaf(out_features=1, out_channels=2),
        make_normal_leaf(out_features=1, out_channels=2),
    ]
    weights = torch.ones((1, 2, 2, 2, 1))
    weights /= weights.sum(dim=3, keepdim=True)
    with pytest.raises(InvalidParameterCombinationError):
        ElementwiseSum(inputs=leaves, weights=weights, num_repetitions=1)
