from __future__ import annotations

import pytest

from tests.contract_data import EINET_LAYER_TYPE_VALUES, EINET_PARAMS_SAMPLING
from tests.test_helpers.assertions import assert_finite_tensor
from tests.test_helpers.builders import make_einet

pytestmark = [pytest.mark.contract, pytest.mark.slow_matrix]


@pytest.mark.parametrize("num_sums,num_leaves,depth,num_reps,layer_type,structure", EINET_PARAMS_SAMPLING)
def test_parametrized_sampling_contract(num_sums, num_leaves, depth, num_reps, layer_type, structure):
    # Sampling traverses the same split hierarchy as forward passes, so it needs the
    # same minimum feature budget for deeper configurations.
    num_features = max(4, 2**depth)
    model = make_einet(
        num_features=num_features,
        num_classes=1,
        num_sums=num_sums,
        num_leaves=num_leaves,
        depth=depth,
        num_repetitions=num_reps,
        layer_type=layer_type,
        structure=structure,
    )

    samples = model.sample(num_samples=20)
    assert samples.shape == (20, num_features)
    assert_finite_tensor(samples)


@pytest.mark.parametrize("layer_type", EINET_LAYER_TYPE_VALUES)
def test_mpe_sampling_contract(layer_type: str):
    model = make_einet(
        num_features=4,
        num_classes=1,
        num_sums=5,
        num_leaves=3,
        depth=1,
        num_repetitions=2,
        layer_type=layer_type,
        structure="top-down",
    )

    # Keep this contract pinned to top-down to limit matrix size; bottom-up MPE
    # coverage is exercised in module-specific regression tests.
    samples = model.sample(num_samples=10, is_mpe=True)
    assert samples.shape == (10, 4)
    assert_finite_tensor(samples)
