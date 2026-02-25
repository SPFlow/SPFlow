"""Sampling and split-related tests for EinsumLayer."""

import pytest

from spflow.modules.einsum import EinsumLayer
from tests.modules.einsum.layer_test_utils import (
    assert_split_alternate_input_differentiable_sampling,
    assert_split_alternate_input_works,
    assert_split_input_not_wrapped,
    assert_split_input_produces_same_output,
    assert_split_sampling_works,
)
from tests.utils.leaves import make_normal_leaf


class TestEinsumLayerConstruction:
    """Construction validation tests not covered by contracts."""

    def test_invalid_odd_features(self):
        # Single-input Einsum internally pairs adjacent features; odd counts cannot be paired.
        inputs = make_normal_leaf(out_features=3, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            EinsumLayer(inputs=inputs, out_channels=2)


class TestEinsumLayerSplitOptimization:
    """Test that EinsumLayer reuses Split modules when passed directly."""

    def test_split_input_not_wrapped(self):
        assert_split_input_not_wrapped(EinsumLayer)

    def test_split_input_produces_same_output(self):
        assert_split_input_produces_same_output(EinsumLayer)

    def test_split_sampling_works(self):
        assert_split_sampling_works(EinsumLayer)

    def test_split_alternate_input_works(self):
        assert_split_alternate_input_works(EinsumLayer)

    def test_split_alternate_input_differentiable_sampling(self):
        assert_split_alternate_input_differentiable_sampling(EinsumLayer)
