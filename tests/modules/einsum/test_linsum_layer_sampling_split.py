"""Sampling and split-related tests for LinsumLayer."""

import pytest

from spflow.modules.einsum import LinsumLayer
from tests.modules.einsum.layer_test_utils import (
    assert_split_alternate_input_differentiable_sampling,
    assert_split_alternate_input_works,
    assert_split_input_not_wrapped,
    assert_split_input_produces_same_output,
    assert_split_sampling_works,
)
from tests.utils.leaves import make_normal_leaf


class TestLinsumLayerConstruction:
    def test_invalid_odd_features(self):
        # Linsum's split path needs even feature counts to build left/right pairs.
        inputs = make_normal_leaf(out_features=3, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            LinsumLayer(inputs=inputs, out_channels=2)


class TestLinsumLayerSplitConsecutiveOptimization:
    def test_split_mode_input_not_wrapped(self):
        assert_split_input_not_wrapped(LinsumLayer)

    def test_split_mode_input_produces_same_output(self):
        assert_split_input_produces_same_output(LinsumLayer)

    def test_split_mode_sampling_works(self):
        assert_split_sampling_works(LinsumLayer)

    def test_split_alternate_input_works(self):
        assert_split_alternate_input_works(LinsumLayer)

    def test_split_alternate_input_differentiable_sampling(self):
        assert_split_alternate_input_differentiable_sampling(LinsumLayer)
