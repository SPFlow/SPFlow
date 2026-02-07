"""Weights, repr, and marginalization tests for EinsumLayer."""

import pytest
import torch

from spflow.modules.einsum import EinsumLayer
from tests.modules.einsum.layer_test_utils import make_einsum_single_input


class TestEinsumLayerWeights:
    def test_log_weights_consistent(self):
        module = make_einsum_single_input(3, 4, 6, 2)
        expected = torch.log(module.weights)
        actual = module.log_weights
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)

    def test_set_weights(self):
        module = make_einsum_single_input(2, 3, 4, 1)
        new_weights = torch.rand(module.weights_shape) + 1e-8
        new_weights = new_weights / new_weights.sum(dim=(-2, -1), keepdim=True)
        module.weights = new_weights
        torch.testing.assert_close(module.weights, new_weights, rtol=1e-5, atol=1e-5)

    def test_set_invalid_weights_shape(self):
        module = make_einsum_single_input(2, 3, 4, 1)
        with pytest.raises(ValueError):
            module.weights = torch.rand(1, 2, 3)

    def test_set_invalid_weights_not_normalized(self):
        module = make_einsum_single_input(2, 3, 4, 1)
        with pytest.raises(ValueError):
            module.weights = torch.rand(module.weights_shape) + 0.1


class TestEinsumLayerMarginalization:
    def test_marginalize_partial_single_input(self):
        module = make_einsum_single_input(2, 3, 4, 1)
        marg_module = module.marginalize([0])
        assert marg_module is not None

    def test_marginalize_full_single_input(self):
        module = make_einsum_single_input(2, 3, 4, 1)
        all_vars = list(module.scope.query)
        marg_module = module.marginalize(all_vars)
        assert marg_module is None

    def test_marginalize_no_overlap(self):
        module = make_einsum_single_input(2, 3, 4, 1)
        marg_module = module.marginalize([100, 101])
        assert isinstance(marg_module, EinsumLayer)


class TestEinsumLayerExtraRepr:
    def test_extra_repr(self):
        module = make_einsum_single_input(2, 3, 4, 1)
        repr_str = module.extra_repr()
        assert "weights=" in repr_str
        assert str(module.weights_shape) in repr_str
