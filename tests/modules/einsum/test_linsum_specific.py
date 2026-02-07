"""LinsumLayer-specific tests not covered by shared contracts."""

import pytest

from spflow.meta import Scope
from spflow.modules.einsum import EinsumLayer, LinsumLayer
from tests.utils.leaves import DummyLeaf, make_leaf, make_normal_leaf


@pytest.mark.contract
def test_linsum_weight_shape_is_linear_vs_einsum() -> None:
    inputs = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
    einsum = EinsumLayer(out_channels=3, inputs=inputs, num_repetitions=1)
    linsum = LinsumLayer(out_channels=3, inputs=inputs, num_repetitions=1)

    assert len(linsum.weights_shape) == 4
    assert len(einsum.weights_shape) == 5


@pytest.mark.contract
def test_linsum_rejects_asymmetric_two_input_channels() -> None:
    left = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0, 1]), num_repetitions=1)
    right = make_leaf(cls=DummyLeaf, out_channels=3, scope=Scope([2, 3]), num_repetitions=1)
    with pytest.raises(ValueError):
        LinsumLayer(inputs=[left, right], out_channels=4, num_repetitions=1)
