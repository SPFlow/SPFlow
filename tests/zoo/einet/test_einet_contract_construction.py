from __future__ import annotations

import pytest

from tests.contract_data import EINET_PARAMS_FULL
from tests.test_helpers.builders import make_einet

pytestmark = [pytest.mark.contract, pytest.mark.slow_matrix]


@pytest.mark.parametrize("num_sums,num_leaves,depth,num_reps,layer_type,structure", EINET_PARAMS_FULL)
def test_parametrized_construction_contract(num_sums, num_leaves, depth, num_reps, layer_type, structure):
    # Binary tree layouts require at least 2**depth features; keep a floor to also
    # exercise shallow settings where depth alone would under-constrain the input size.
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

    assert model.num_features == num_features
    assert model.num_sums == num_sums
    assert model.num_leaves == num_leaves
    assert model.depth == depth
    assert model.num_repetitions == num_reps
    assert model.layer_type == layer_type
    assert model.structure == structure


@pytest.mark.parametrize("num_classes", [1, 3, 5])
def test_multi_class_construction_contract(num_classes: int):
    model = make_einet(
        num_features=4,
        num_classes=num_classes,
        num_sums=5,
        num_leaves=3,
        depth=1,
        num_repetitions=2,
    )
    # Constructor should preserve class cardinality exactly since downstream posterior
    # and sampling paths branch on this value.
    assert model.num_classes == num_classes
