from itertools import product

import pytest

from spflow.learn import expectation_maximization
from tests.modules.module_contract_data import PRODUCT_PARAMS
from tests.modules.test_helpers.builders import build_product
from tests.utils.leaves import make_normal_data

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("in_channels,out_features,num_reps", PRODUCT_PARAMS)
def test_product_em_contract(in_channels: int, out_features: int, num_reps: int):
    module = build_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    expectation_maximization(module, make_normal_data(out_features=out_features), max_steps=2)


@pytest.mark.parametrize(
    "prune,in_channels,marg_rvs,num_reps",
    product(
        [True, False],
        [1, 3],
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        [1, 5],
    ),
)
def test_product_marginalization_contract(prune: bool, in_channels: int, marg_rvs: list[int], num_reps: int):
    out_features = 3
    module = build_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    marginalized = module.marginalize(marg_rvs, prune=prune)
    if len(marg_rvs) == out_features:
        assert marginalized is None
    else:
        assert marginalized is not None
