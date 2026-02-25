"""EinsumLayer-specific tests not covered by shared contracts."""

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.einsum import EinsumLayer
from tests.utils.leaves import DummyLeaf, make_leaf, make_normal_data


@pytest.mark.contract
def test_einsum_two_inputs_support_asymmetric_channels_end_to_end() -> None:
    left = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0, 1]), num_repetitions=1)
    right = make_leaf(cls=DummyLeaf, out_channels=3, scope=Scope([2, 3]), num_repetitions=1)
    module = EinsumLayer(inputs=[left, right], out_channels=4)

    assert module.weights_shape == (2, 4, 1, 2, 3)
    lls = module.log_likelihood(make_normal_data(num_samples=12, out_features=4))
    assert lls.shape == (12, 2, 4, 1)
    assert torch.isfinite(lls).all()


@pytest.mark.contract
def test_einsum_two_inputs_requires_disjoint_scopes() -> None:
    left = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0, 1]), num_repetitions=1)
    right = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([1, 2]), num_repetitions=1)
    # Overlapping scopes would violate decomposability for product-like composition.
    with pytest.raises(ValueError):
        EinsumLayer(inputs=[left, right], out_channels=2)
