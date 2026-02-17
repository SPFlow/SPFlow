"""Focused tests for Integral module branches."""

import pytest
import torch

from spflow.meta.data.scope import Scope
from spflow.zoo.pic.integral import Integral
from tests.utils.leaves import DummyLeaf, make_leaf


@pytest.fixture
def dummy_input_module():
    return make_leaf(cls=DummyLeaf, out_channels=2, out_features=3, num_repetitions=1)


def test_integral_initialization_shapes_and_scope(dummy_input_module) -> None:
    module = Integral(
        input_module=dummy_input_module,
        latent_scope=Scope([10]),
        integrated_latent_scope=[11, 12],
        function=None,
        function_head_idx=None,
    )

    assert module.inputs is dummy_input_module
    assert module.scope == dummy_input_module.scope
    assert module.in_shape == dummy_input_module.out_shape
    assert module.out_shape.features == module.in_shape.features
    assert module.out_shape.channels == 1
    assert module.out_shape.repetitions == 1
    assert module.feature_to_scope.shape == dummy_input_module.feature_to_scope.shape


def test_integral_not_implemented_methods(dummy_input_module) -> None:
    module = Integral(
        input_module=dummy_input_module,
        latent_scope=None,
        integrated_latent_scope=None,
        function=None,
        function_head_idx=0,
    )

    data = torch.randn(5, len(module.scope.query))
    with pytest.raises(NotImplementedError):
        module.log_likelihood(data)
    with pytest.raises(NotImplementedError):
        module.sample(num_samples=3)
    with pytest.raises(NotImplementedError):
        module.marginalize([0])
