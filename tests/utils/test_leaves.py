"""Tests for leaf distribution utility functions."""

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.utils.leaves import (
    _handle_mle_edge_cases,
    apply_nan_strategy,
    init_parameter,
    parse_leaf_args,
    validate_all_or_none,
)


class TestValidateAllOrNone:
    """Test validate_all_or_none function."""

    def test_validate_all_or_none_all_none(self):
        """Test validate_all_or_none when all params are None."""
        # All None is valid
        result = validate_all_or_none(a=None, b=None, c=None)
        assert result is False  # Returns False when none provided

    def test_validate_all_or_none_all_provided(self):
        """Test validate_all_or_none when all params provided."""
        # All provided is valid
        result = validate_all_or_none(a=1.0, b=2.0, c=3.0)
        assert result is True  # Returns True when any provided

    def test_validate_all_or_none_partial_params(self):
        """Test validate_all_or_none with partial params."""
        # Partial params should raise error
        with pytest.raises(InvalidParameterCombinationError) as exc_info:
            validate_all_or_none(a=1.0, b=None, c=3.0)

        # Check error message is informative
        assert "must be provided together" in str(exc_info.value)
        assert "a, b, c" in str(exc_info.value)

    def test_validate_all_or_none_single_missing(self):
        """Test with only one parameter missing."""
        with pytest.raises(InvalidParameterCombinationError):
            validate_all_or_none(mean=torch.tensor([1.0]), std=None)

    def test_validate_all_or_none_zero_as_value(self):
        """Test that 0 is treated as a value, not None."""
        # 0 is a valid value, not None
        with pytest.raises(InvalidParameterCombinationError):
            validate_all_or_none(a=0, b=None)

    def test_validate_all_or_none_false_as_value(self):
        """Test that False is treated as a value, not None."""
        # False is a valid value, not None
        with pytest.raises(InvalidParameterCombinationError):
            validate_all_or_none(a=False, b=None)

    def test_validate_all_or_none_empty_string_as_value(self):
        """Test that empty string is treated as a value, not None."""
        # Empty string is a valid value, not None
        with pytest.raises(InvalidParameterCombinationError):
            validate_all_or_none(a="", b=None)

    def test_validate_all_or_none_two_params(self):
        """Test with two parameters."""
        # Both None
        assert validate_all_or_none(loc=None, scale=None) is False
        # Both provided
        assert validate_all_or_none(loc=1.0, scale=2.0) is True
        # One provided
        with pytest.raises(InvalidParameterCombinationError):
            validate_all_or_none(loc=1.0, scale=None)


class TestApplyNanStrategy:
    """Test apply_nan_strategy function."""

    def test_apply_nan_strategy_ignore_all_valid(self, device):
        """Test apply_nan_strategy with 'ignore' and all valid data."""
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        weights = torch.ones((data.shape[0], data.shape[1], 1, 1), device=device)

        result_data, result_weights = apply_nan_strategy("ignore", data, device, weights)

        # Data should be unchanged
        assert torch.equal(result_data, data)
        # Weights should keep shape and normalize to number of samples
        assert result_weights.shape == weights.shape
        assert torch.allclose(result_weights.sum(), torch.tensor(float(data.shape[0]), device=device))

    def test_apply_nan_strategy_ignore_some_nan(self, device):
        """Test apply_nan_strategy with 'ignore' and some NaN."""
        data = torch.tensor([[1.0, 2.0], [float("nan"), 4.0], [5.0, 6.0]], device=device)
        weights = torch.ones((data.shape[0], data.shape[1], 1, 1), device=device)

        result_data, result_weights = apply_nan_strategy("ignore", data, device, weights)

        # Should remove row with NaN
        expected_data = torch.tensor([[1.0, 2.0], [5.0, 6.0]], device=device)
        assert torch.equal(result_data, expected_data)
        # Weights should be adjusted
        assert result_weights.shape == (expected_data.shape[0], expected_data.shape[1], 1, 1)
        assert torch.allclose(
            result_weights.sum(), torch.tensor(float(expected_data.shape[0]), device=device)
        )

    def test_apply_nan_strategy_ignore_all_nan(self, device):
        """Test apply_nan_strategy with 'ignore' and all NaN."""
        data = torch.tensor([[float("nan"), float("nan")], [float("nan"), float("nan")]], device=device)
        weights = torch.ones((data.shape[0], data.shape[1], 1, 1), device=device)

        with pytest.raises(ValueError) as exc_info:
            apply_nan_strategy("ignore", data, device, weights)

        assert "all data is NaN" in str(exc_info.value)

    def test_apply_nan_strategy_with_weights(self, device):
        """Test apply_nan_strategy filters weights consistently."""
        data = torch.tensor([[1.0, 2.0], [float("nan"), 4.0], [5.0, 6.0]], device=device)
        weights = torch.tensor(
            [
                [[[1.0]], [[1.0]]],
                [[[2.0]], [[2.0]]],
                [[[3.0]], [[3.0]]],
            ],
            device=device,
        )

        result_data, result_weights = apply_nan_strategy("ignore", data, device, weights)

        # Should keep rows 0 and 2 (weights 1.0 and 3.0)
        expected_data = torch.tensor([[1.0, 2.0], [5.0, 6.0]], device=device)
        assert torch.equal(result_data, expected_data)
        kept_weights = torch.stack([weights[0], weights[2]])
        expected_weights = kept_weights * (expected_data.shape[0] / kept_weights.sum())
        assert result_weights.shape == expected_weights.shape
        assert torch.allclose(result_weights, expected_weights)

    def test_apply_nan_strategy_weights_shape_error(self, device):
        """Test apply_nan_strategy errors with wrong weights shape."""
        data = torch.tensor([[1.0, 2.0]] * 100, device=device)  # (100, 2)
        weights = torch.ones((50, data.shape[1], 1, 1), device=device)  # Wrong first dimension

        with pytest.raises(ValueError) as exc_info:
            apply_nan_strategy(None, data, device, weights)

        assert "Weights shape" in str(exc_info.value)
        assert "does not match" in str(exc_info.value)

    def test_apply_nan_strategy_handles_multidimensional_weights(self, device):
        """Test apply_nan_strategy accepts multi-dimensional weights."""
        data = torch.tensor([[1.0, 2.0]] * 10, device=device)  # (10, 2)
        weights = torch.ones((data.shape[0], data.shape[1], 1, 1), device=device)

        result_data, result_weights = apply_nan_strategy(None, data, device, weights)

        assert torch.equal(result_data, data)
        assert result_weights.shape == weights.shape

    def test_apply_nan_strategy_none_with_nan_error(self, device):
        """Test apply_nan_strategy errors when strategy is None and data has NaN."""
        data = torch.tensor([[1.0, 2.0], [float("nan"), 4.0]], device=device)
        weights = torch.ones((data.shape[0], data.shape[1], 1, 1), device=device)

        with pytest.raises(ValueError) as exc_info:
            apply_nan_strategy(None, data, device, weights)

        assert "missing (NaN) values" in str(exc_info.value)
        assert "nan_strategy" in str(exc_info.value)

    def test_apply_nan_strategy_unknown_strategy(self, device):
        """Test apply_nan_strategy with unknown strategy."""
        data = torch.tensor([[1.0, 2.0], [float("nan"), 4.0]], device=device)
        weights = torch.ones((data.shape[0], data.shape[1], 1, 1), device=device)

        with pytest.raises(ValueError) as exc_info:
            apply_nan_strategy("unknown", data, device, weights)

        assert "Unknown nan_strategy" in str(exc_info.value)

    def test_apply_nan_strategy_weights_normalization(self, device):
        """Test that weights are normalized to sum to number of samples."""
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        weights = torch.tensor(
            [
                [[[1.0]], [[2.0]]],
                [[[3.0]], [[4.0]]],
                [[[5.0]], [[6.0]]],
            ],
            device=device,
        )

        result_data, result_weights = apply_nan_strategy(None, data, device, weights)

        # Weights should sum to number of samples (3)
        assert torch.allclose(result_weights.sum(), torch.tensor(3.0, device=device))

    def test_apply_nan_strategy_nan_in_different_columns(self, device):
        """Test with NaN in different columns per row."""
        data = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, float("nan"), 6.0], [7.0, 8.0, float("nan")]], device=device
        )
        weights = torch.ones((data.shape[0], data.shape[1], 1, 1), device=device)

        result_data, result_weights = apply_nan_strategy("ignore", data, device, weights)

        # Should only keep first row (no NaN)
        expected_data = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        assert torch.equal(result_data, expected_data)
        assert result_weights.shape == (expected_data.shape[0], expected_data.shape[1], 1, 1)


class TestInitParameter:
    """Test init_parameter function."""

    def test_init_parameter_basic(self, device):
        """Test init_parameter with valid parameters."""
        event_shape = (10, 5)

        def init_fn(shape):
            return torch.ones(shape, device=device)

        result = init_parameter(None, event_shape, init_fn)

        assert result.shape == event_shape
        assert torch.all(result == 1.0)

    def test_init_parameter_with_provided_param(self, device):
        """Test init_parameter with already provided parameter."""
        provided_param = torch.tensor([1.0, 2.0, 3.0], device=device)

        def init_fn(shape):
            return torch.zeros(shape, device=device)

        result = init_parameter(provided_param, (3,), init_fn)

        # Should return provided param, not call init_fn
        assert torch.equal(result, provided_param)

    def test_init_parameter_with_nan(self, device):
        """Test init_parameter with NaN value."""
        nan_param = torch.tensor([1.0, float("nan"), 3.0], device=device)

        def init_fn(shape):
            return torch.zeros(shape, device=device)

        with pytest.raises(ValueError) as exc_info:
            init_parameter(nan_param, (3,), init_fn)

        assert "must be finite" in str(exc_info.value)

    def test_init_parameter_with_inf(self, device):
        """Test init_parameter with infinity."""
        inf_param = torch.tensor([1.0, float("inf"), 3.0], device=device)

        def init_fn(shape):
            return torch.zeros(shape, device=device)

        with pytest.raises(ValueError) as exc_info:
            init_parameter(inf_param, (3,), init_fn)

        assert "must be finite" in str(exc_info.value)

    def test_init_parameter_with_neg_inf(self, device):
        """Test init_parameter with negative infinity."""
        neg_inf_param = torch.tensor([1.0, float("-inf"), 3.0], device=device)

        def init_fn(shape):
            return torch.zeros(shape, device=device)

        with pytest.raises(ValueError):
            init_parameter(neg_inf_param, (3,), init_fn)

    def test_init_parameter_batch_shape(self, device):
        """Test init_parameter with batch dimensions."""
        event_shape = (32, 10, 5)

        def init_fn(shape):
            return torch.randn(shape, device=device)

        result = init_parameter(None, event_shape, init_fn)

        assert result.shape == event_shape


class TestParseLeafArgs:
    """Test parse_leaf_args function."""

    def test_parse_leaf_args_with_scope(self):
        """Test parse_leaf_args with Scope input."""
        scope = Scope([0, 1, 2])
        out_channels = 5
        num_repetitions = None
        params = None

        event_shape = parse_leaf_args(scope, out_channels, num_repetitions, params)

        # Should return (query_length, out_channels)
        assert event_shape == (3, 5)

    def test_parse_leaf_args_with_int(self):
        """Test parse_leaf_args with integer input."""
        scope = 5  # int
        out_channels = 10
        num_repetitions = None
        params = None

        event_shape = parse_leaf_args(scope, out_channels, num_repetitions, params)

        # Should return (1, out_channels) for int scope
        assert event_shape == (1, 10)

    def test_parse_leaf_args_with_list(self):
        """Test parse_leaf_args with list input."""
        scope = [0, 1, 2, 3]  # list
        out_channels = 7
        num_repetitions = None
        params = None

        event_shape = parse_leaf_args(scope, out_channels, num_repetitions, params)

        # Should return (len(list), out_channels)
        assert event_shape == (4, 7)

    def test_parse_leaf_args_with_repetitions(self):
        """Test parse_leaf_args with num_repetitions."""
        scope = Scope([0, 1])
        out_channels = 5
        num_repetitions = 3
        params = None

        event_shape = parse_leaf_args(scope, out_channels, num_repetitions, params)

        # Should return (query_length, out_channels, num_repetitions)
        assert len(event_shape) == 3
        assert event_shape == (2, 5, 3)

    def test_parse_leaf_args_with_params(self):
        """Test parse_leaf_args with provided parameters."""
        scope = Scope([0, 1, 2])
        out_channels = None
        num_repetitions = None
        param1 = torch.randn(3, 5)
        param2 = torch.randn(3, 5)
        params = [param1, param2]

        event_shape = parse_leaf_args(scope, out_channels, num_repetitions, params)

        # Should infer from params
        assert event_shape == (3, 5)

    def test_parse_leaf_args_params_mismatch_scope(self):
        """Test parse_leaf_args with params that don't match scope."""
        scope = Scope([0, 1, 2])  # 3 features
        out_channels = None
        num_repetitions = None
        param1 = torch.randn(5, 10)  # Wrong first dimension
        params = [param1]

        with pytest.raises(InvalidParameterCombinationError) as exc_info:
            parse_leaf_args(scope, out_channels, num_repetitions, params)

        assert "scope dimensions" in str(exc_info.value)

    def test_parse_leaf_args_params_mismatch_out_channels(self):
        """Test parse_leaf_args with params that don't match out_channels."""
        scope = Scope([0, 1])
        out_channels = 5
        num_repetitions = None
        param1 = torch.randn(2, 10)  # Second dimension doesn't match out_channels
        params = [param1]

        with pytest.raises(InvalidParameterCombinationError) as exc_info:
            parse_leaf_args(scope, out_channels, num_repetitions, params)

        assert "out_channels" in str(exc_info.value)

    def test_parse_leaf_args_partial_params_error(self):
        """Test parse_leaf_args with partial params (some None, some not)."""
        scope = Scope([0, 1])
        out_channels = 5
        num_repetitions = None
        param1 = torch.randn(2, 5)
        params = [param1, None]  # Partial params

        with pytest.raises(InvalidParameterCombinationError) as exc_info:
            parse_leaf_args(scope, out_channels, num_repetitions, params)

        assert "all parameters or none" in str(exc_info.value)

    def test_parse_leaf_args_no_out_channels_no_params(self):
        """Test parse_leaf_args with neither out_channels nor params."""
        scope = Scope([0, 1])
        out_channels = None
        num_repetitions = None
        params = None

        with pytest.raises(InvalidParameterCombinationError) as exc_info:
            parse_leaf_args(scope, out_channels, num_repetitions, params)

        assert "out_channels or distribution parameters" in str(exc_info.value)

    def test_parse_leaf_args_invalid_scope_type(self):
        """Test parse_leaf_args with invalid scope type."""
        scope = "invalid"  # Invalid type
        out_channels = 5
        num_repetitions = None
        params = None

        with pytest.raises(ValueError) as exc_info:
            parse_leaf_args(scope, out_channels, num_repetitions, params)

        assert "scope must be of type" in str(exc_info.value)


class TestHandleMleEdgeCases:
    """Test _handle_mle_edge_cases function."""

    def test_handle_mle_edge_cases_clean_data(self, device):
        """Test _handle_mle_edge_cases with clean data."""
        param_est = torch.tensor([1.0, 2.0, 3.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=0.0, ub=10.0)

        # Should be unchanged
        assert torch.equal(result, param_est)

    def test_handle_mle_edge_cases_with_nan(self, device):
        """Test _handle_mle_edge_cases with NaN values."""
        param_est = torch.tensor([1.0, float("nan"), 3.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=0.0, ub=10.0)

        # NaN should be replaced with eps (1e-8)
        assert not torch.isnan(result).any()
        assert result[1] == pytest.approx(1e-8, abs=1e-9)

    def test_handle_mle_edge_cases_below_lower_bound(self, device):
        """Test _handle_mle_edge_cases with values below lower bound."""
        param_est = torch.tensor([1.0, -0.5, 3.0], device=device)
        lb = 0.0
        result = _handle_mle_edge_cases(param_est, lb=lb, ub=10.0)

        # Value below lb should be clamped to lb + eps
        assert result[1] > lb
        assert result[1] == pytest.approx(lb + 1e-8, abs=1e-9)

    def test_handle_mle_edge_cases_above_upper_bound(self, device):
        """Test _handle_mle_edge_cases with values above upper bound."""
        param_est = torch.tensor([1.0, 15.0, 3.0], device=device)
        ub = 10.0
        result = _handle_mle_edge_cases(param_est, lb=0.0, ub=ub)

        # Value above ub should be clamped to ub - eps
        assert result[1] <= ub
        assert result[1] == pytest.approx(ub - 1e-8, abs=1e-9)

    def test_handle_mle_edge_cases_at_lower_bound(self, device):
        """Test _handle_mle_edge_cases with values at lower bound."""
        lb = 0.0
        param_est = torch.tensor([1.0, lb, 3.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=lb, ub=10.0)

        # Value at lb should be adjusted to lb + eps
        assert result[1] > lb

    def test_handle_mle_edge_cases_at_upper_bound(self, device):
        """Test _handle_mle_edge_cases with values at upper bound."""
        ub = 10.0
        param_est = torch.tensor([1.0, ub, 3.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=0.0, ub=ub)

        # Value at ub should be adjusted to ub - eps
        assert result[1] <= ub

    def test_handle_mle_edge_cases_zero_without_bounds(self, device):
        """Test _handle_mle_edge_cases with zero values when no bounds."""
        param_est = torch.tensor([1.0, 0.0, 3.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=None, ub=None)

        # Zero should be replaced with eps when no lower bound
        assert result[1] > 0
        assert result[1] == pytest.approx(1e-8, abs=1e-9)

    def test_handle_mle_edge_cases_tensor_bounds(self, device):
        """Test _handle_mle_edge_cases with tensor bounds."""
        param_est = torch.tensor([1.0, 2.0, 3.0], device=device)
        lb = torch.tensor([0.0, 1.5, 0.0], device=device)
        ub = torch.tensor([5.0, 10.0, 2.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=lb, ub=ub)

        # Third value (3.0) is above its upper bound (2.0)
        assert result[2] <= ub[2]

    def test_handle_mle_edge_cases_only_lower_bound(self, device):
        """Test _handle_mle_edge_cases with only lower bound."""
        param_est = torch.tensor([1.0, -0.5, 3.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=0.0, ub=None)

        # Value below lb should be adjusted
        assert result[1] > 0

    def test_handle_mle_edge_cases_only_upper_bound(self, device):
        """Test _handle_mle_edge_cases with only upper bound."""
        param_est = torch.tensor([1.0, 15.0, 3.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=None, ub=10.0)

        # Value above ub should be adjusted
        assert result[1] <= 10.0


class TestLeafUtilsIntegration:
    """Integration tests for leaf utilities."""

    def test_leaves_utils_parameter_flow(self, device):
        """Test parameter flow through utilities."""
        # Initialize parameter
        event_shape = (5, 10)

        def init_fn(shape):
            return torch.ones(shape, device=device) * 0.5

        param = init_parameter(None, event_shape, init_fn)

        # Validate
        result = validate_all_or_none(param1=param, param2=param)
        assert result is True

    def test_leaves_utils_nan_handling_flow(self, device):
        """Test NaN handling flow."""
        # Create data with NaN
        data = torch.tensor([[1.0, 2.0], [float("nan"), 4.0], [5.0, 6.0]], device=device)
        weights = torch.ones((data.shape[0], data.shape[1], 1, 1), device=device)

        # Apply NaN strategy
        clean_data, weights = apply_nan_strategy("ignore", data, device, weights)

        # Handle edge cases in estimates
        estimates = clean_data.mean(dim=0)
        final = _handle_mle_edge_cases(estimates, lb=0.0, ub=10.0)

        assert final.shape == (2,)
        assert not torch.isnan(final).any()

    def test_leaves_utils_with_gradients(self, device):
        """Test gradient flow through parameters."""
        event_shape = (5, 10)

        def init_fn(shape):
            return torch.randn(shape, device=device, requires_grad=True)

        param = init_parameter(None, event_shape, init_fn)

        # Should have gradients enabled
        loss = param.sum()
        loss.backward()
        assert param.grad is not None


class TestLeafUtilsEdgeCases:
    """Test edge cases in leaf utilities."""

    def test_handle_mle_multiple_edge_cases(self, device):
        """Test handling multiple edge cases at once."""
        param_est = torch.tensor([float("nan"), -1.0, 5.0, 15.0, 0.0], device=device)
        result = _handle_mle_edge_cases(param_est, lb=0.0, ub=10.0)

        # NaN -> eps, -1 -> eps, 5 -> 5, 15 -> 10-eps, 0 -> eps
        assert not torch.isnan(result).any()
        assert torch.all(result > 0)
        assert torch.all(result <= 10.0)
        assert result[2] == pytest.approx(5.0, abs=1e-6)  # 5 is in bounds

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_handle_mle_dtype_preservation(self, device, dtype):
        """Test dtype preservation in edge case handling."""
        param_est = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        result = _handle_mle_edge_cases(param_est, lb=0.0, ub=10.0)

        assert result.dtype == dtype
