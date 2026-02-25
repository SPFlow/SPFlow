"""Weights, repr, and marginalization tests for LinsumLayer."""

import pytest
import torch

from tests.modules.einsum.layer_test_utils import (
    assert_extra_repr_contains_weights_shape,
    assert_log_weights_consistent,
    assert_marginalize_full_single_input,
    assert_marginalize_partial_single_input,
    assert_set_weights_round_trip,
    make_linsum_single_input,
)


class TestLinsumLayerWeights:
    def test_log_weights_consistent(self):
        assert_log_weights_consistent(make_linsum_single_input)

    def test_set_weights(self):
        assert_set_weights_round_trip(make_linsum_single_input, normalize_dims=(-1,))

    def test_set_invalid_weights_shape(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        with pytest.raises(ValueError):
            module.weights = torch.rand(1, 2, 3)

    def test_set_invalid_weights_not_normalized(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        with pytest.raises(ValueError):
            module.weights = torch.rand(module.weights_shape) + 0.1


class TestLinsumLayerMarginalization:
    def test_marginalize_partial_single_input(self):
        assert_marginalize_partial_single_input(make_linsum_single_input)

    def test_marginalize_full_single_input(self):
        assert_marginalize_full_single_input(make_linsum_single_input)

    def test_marginalize_no_overlap(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        # Disjoint marginalization requests should be a no-op, not a structural rewrite.
        marg_module = module.marginalize([100, 101])
        assert marg_module is not None


class TestLinsumLayerExtraRepr:
    def test_extra_repr(self):
        assert_extra_repr_contains_weights_shape(make_linsum_single_input)
