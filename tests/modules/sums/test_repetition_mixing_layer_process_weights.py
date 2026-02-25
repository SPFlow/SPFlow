"""Tests for RepetitionMixingLayer._process_weights_parameter method."""

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.sums import RepetitionMixingLayer


def _rand(*size: int) -> torch.Tensor:
    return torch.rand(*size)


class TestProcessWeightsParameter:
    """Tests for the _process_weights_parameter method of RepetitionMixingLayer."""

    @pytest.fixture
    def layer(self):
        """Create a minimal RepetitionMixingLayer instance for method testing."""
        # We call a protected helper directly, so we build the lightest valid instance.
        from spflow.modules.leaves import Normal

        leaf = Normal(
            scope=Scope([0, 1, 2]),
            out_channels=4,
            num_repetitions=3,
        )
        return RepetitionMixingLayer(inputs=leaf, out_channels=4, num_repetitions=3)

    def test_none_weights_passthrough(self, layer):
        """Test that None weights returns unchanged values."""
        weights, out_channels, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=None,
            out_channels=5,
            num_repetitions=3,
        )
        assert weights is None
        assert out_channels == 5
        assert num_repetitions == 3

    def test_1d_weights_reshaped_to_3d(self, layer):
        """Test that 1D weights are reshaped to (1, -1, 1)."""
        input_weights = torch.tensor([0.25, 0.25, 0.25, 0.25])
        weights, out_channels, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert weights.dim() == 3
        assert weights.shape == (1, 4, 1)
        assert out_channels == 4
        assert num_repetitions == 1

    def test_2d_weights_reshaped_to_3d(self, layer):
        """Test that 2D weights are reshaped to (1, shape[0], shape[1])."""
        input_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]])
        weights, out_channels, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert weights.dim() == 3
        assert weights.shape == (1, 3, 2)
        assert out_channels == 3
        assert num_repetitions == 2

    def test_3d_weights_unchanged(self, layer):
        """Test that 3D weights pass through unchanged."""
        input_weights = _rand(2, 4, 3)
        weights, out_channels, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert weights.dim() == 3
        assert weights.shape == (2, 4, 3)
        assert out_channels == 4
        assert num_repetitions == 3

    def test_out_channels_inferred_from_1d_weights(self, layer):
        """Test that out_channels is inferred from 1D weights."""
        input_weights = torch.tensor([0.2, 0.3, 0.5])
        _, out_channels, _ = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert out_channels == 3

    def test_out_channels_inferred_from_2d_weights(self, layer):
        """Test that out_channels is inferred from 2D weights (dim 0)."""
        input_weights = _rand(5, 2)
        _, out_channels, _ = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert out_channels == 5

    def test_out_channels_inferred_from_3d_weights(self, layer):
        """Test that out_channels is inferred from 3D weights (dim 1)."""
        input_weights = _rand(3, 7, 2)
        _, out_channels, _ = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert out_channels == 7

    def test_num_repetitions_inferred_from_1d_weights(self, layer):
        """Test that num_repetitions defaults to 1 for 1D weights."""
        input_weights = torch.tensor([0.2, 0.3, 0.5])
        _, _, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert num_repetitions == 1

    def test_num_repetitions_inferred_from_2d_weights(self, layer):
        """Test that num_repetitions is inferred from 2D weights (dim 1)."""
        input_weights = _rand(5, 4)
        _, _, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert num_repetitions == 4

    def test_num_repetitions_inferred_from_3d_weights(self, layer):
        """Test that num_repetitions is inferred from 3D weights (dim -1)."""
        input_weights = _rand(2, 5, 6)
        _, _, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        assert num_repetitions == 6

    def test_error_when_out_channels_specified_with_weights(self, layer):
        """Test that specifying both out_channels and weights raises error."""
        input_weights = _rand(4)
        with pytest.raises(InvalidParameterCombinationError):
            layer._process_weights_parameter(
                inputs=None,
                weights=input_weights,
                out_channels=5,
                num_repetitions=None,
            )

    def test_error_when_weights_dimension_is_4d(self, layer):
        """Test that 4D weights raise a ValueError."""
        input_weights = _rand(2, 3, 4, 5)
        with pytest.raises(ValueError):
            layer._process_weights_parameter(
                inputs=None,
                weights=input_weights,
                out_channels=1,
                num_repetitions=None,
            )

    def test_error_when_weights_dimension_is_5d(self, layer):
        """Test that 5D weights raise a ValueError."""
        input_weights = _rand(2, 3, 4, 5, 6)
        with pytest.raises(ValueError):
            layer._process_weights_parameter(
                inputs=None,
                weights=input_weights,
                out_channels=1,
                num_repetitions=None,
            )

    def test_error_when_num_repetitions_mismatches_weights(self, layer):
        """Test that conflicting num_repetitions raises error."""
        input_weights = _rand(3, 4)
        with pytest.raises(InvalidParameterCombinationError):
            layer._process_weights_parameter(
                inputs=None,
                weights=input_weights,
                out_channels=1,
                num_repetitions=2,
            )

    def test_num_repetitions_1_always_allowed(self, layer):
        """Test that num_repetitions=1 is always allowed (special case)."""
        input_weights = _rand(3, 4)
        weights, out_channels, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=1,
        )
        # This compatibility exception keeps older call sites from hard-failing.
        assert num_repetitions == 4

    def test_num_repetitions_matching_is_allowed(self, layer):
        """Test that matching num_repetitions is allowed."""
        input_weights = _rand(3, 4)
        weights, out_channels, num_repetitions = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=4,
        )
        assert num_repetitions == 4

    def test_weight_values_preserved_after_reshape(self, layer):
        """Test that weight values are preserved after reshaping."""
        input_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        weights, _, _ = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        # Reshaping must be metadata-only; probabilities cannot be altered.
        torch.testing.assert_close(weights.squeeze(), input_weights, rtol=0.0, atol=0.0)

    def test_2d_weight_values_preserved_after_reshape(self, layer):
        """Test that 2D weight values are preserved after reshaping."""
        input_weights = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5]])
        weights, _, _ = layer._process_weights_parameter(
            inputs=None,
            weights=input_weights,
            out_channels=1,
            num_repetitions=None,
        )
        # Same invariant for 2D inputs when a feature axis is inserted.
        torch.testing.assert_close(weights.squeeze(0), input_weights, rtol=0.0, atol=0.0)
