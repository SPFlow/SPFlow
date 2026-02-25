"""Tests for tensorized base layer abstractions."""

import pytest
import torch
from torch import Tensor, nn

from spflow.zoo.pic.tensorized.base import TensorizedLayer


class _ToyTensorizedLayer(TensorizedLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight = nn.Parameter(torch.full((self.num_folds, self.num_output_units), 0.5))

    def forward(self, x: Tensor) -> Tensor:
        # Keep implementation trivial so base-class invariants are the only moving part.
        return x.sum(dim=1)


def test_tensorized_layer_constructor_guards() -> None:
    with pytest.raises(ValueError):
        _ToyTensorizedLayer(num_input_units=0, num_output_units=2)
    with pytest.raises(ValueError):
        _ToyTensorizedLayer(num_input_units=2, num_output_units=0)
    with pytest.raises(ValueError):
        _ToyTensorizedLayer(num_input_units=2, num_output_units=2, arity=0)
    with pytest.raises(ValueError):
        _ToyTensorizedLayer(num_input_units=2, num_output_units=2, num_folds=0)


def test_tensorized_layer_num_params_and_reset() -> None:
    layer = _ToyTensorizedLayer(num_input_units=3, num_output_units=4, arity=2, num_folds=2)
    # num_params should reflect registered trainable tensors, not constructor metadata.
    assert layer.num_params == layer.weight.numel()

    with torch.no_grad():
        layer.weight.fill_(5.0)
    layer.reset_parameters()
    assert torch.all(layer.weight >= 0.01)
    assert torch.all(layer.weight <= 0.99)


def test_tensorized_layer_forward_shape() -> None:
    layer = _ToyTensorizedLayer(num_input_units=3, num_output_units=3, arity=2, num_folds=1)
    x = torch.randn(1, 2, 3, 5)
    out = layer(x)
    assert out.shape == (1, 3, 5)
