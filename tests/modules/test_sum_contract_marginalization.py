from itertools import product

import pytest

from tests.modules.module_contract_data import (
    IN_CHANNELS_VALUES,
    OUT_CHANNELS_VALUES,
    NUM_REPETITIONS_VALUES,
)
from tests.modules.test_helpers.builders import build_elementwise_sum, build_sum

pytestmark = pytest.mark.contract

MARG_RVS_VALUES = [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]]


@pytest.mark.parametrize(
    "in_channels,out_channels,marg_rvs,num_reps",
    product(IN_CHANNELS_VALUES, OUT_CHANNELS_VALUES, MARG_RVS_VALUES, NUM_REPETITIONS_VALUES),
)
@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_marginalize_contract(
    in_channels: int, out_channels: int, marg_rvs: list[int], num_reps: int, builder
):
    out_features = 3
    module = builder(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    marginalized = module.marginalize(marg_rvs, prune=False)
    if len(marg_rvs) == out_features:
        assert marginalized is None
        return
    assert marginalized is not None
    assert len(set(marginalized.scope.query).intersection(marg_rvs)) == 0
