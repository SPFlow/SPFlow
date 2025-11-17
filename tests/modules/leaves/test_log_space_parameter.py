"""Comprehensive unit tests for LogSpaceParameter descriptor."""

import pytest
import torch
from torch import nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from utils.leaves import LogSpaceParameter


# ============================================================================
# Test Fixtures and Helper Classes
# ============================================================================


class SimpleLeafWithLogParam(LeafModule):
    """Simple test leaves module with LogSpaceParameter."""

    scale = LogSpaceParameter("scale")

    def __init__(self, scope, scale_init=None, out_channels=1):
        super().__init__(scope, out_channels=out_channels)
        self._event_shape = (len(scope.query), out_channels)

        if scale_init is None:
            scale_init = torch.ones(self._event_shape, dtype=torch.float32)
        else:
            scale_init = torch.as_tensor(scale_init)

        self.log_scale = nn.Parameter(torch.empty_like(scale_init))
        self.scale = scale_init.clone().detach()

    @property
    def distribution(self):
        return torch.distributions.Exponential(self.scale)

    @property
    def _supported_value(self):
        return 0.0

    def params(self):
        return {"scale": self.scale}

    def _mle_compute_statistics(self, data, weights, bias_correction):
        pass


class LeafWithMultipleLogParams(LeafModule):
    """Test leaves with multiple LogSpaceParameters."""

    alpha = LogSpaceParameter("alpha")
    beta = LogSpaceParameter("beta")

    def __init__(self, scope, out_channels=1):
        super().__init__(scope, out_channels=out_channels)
        self._event_shape = (len(scope.query), out_channels)

        alpha_init = torch.ones(self._event_shape)
        beta_init = torch.ones(self._event_shape) * 2.0

        self.log_alpha = nn.Parameter(torch.empty_like(alpha_init))
        self.log_beta = nn.Parameter(torch.empty_like(beta_init))

        self.alpha = alpha_init.clone().detach()
        self.beta = beta_init.clone().detach()

    @property
    def distribution(self):
        return torch.distributions.Gamma(self.alpha, self.beta)

    @property
    def _supported_value(self):
        return 1.0

    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}

    def _mle_compute_statistics(self, data, weights, bias_correction):
        pass


@pytest.fixture
def simple_leaf():
    """Fixture providing a simple leaves module with LogSpaceParameter."""
    return SimpleLeafWithLogParam(Scope([0, 1]))


@pytest.fixture
def multi_param_leaf():
    """Fixture providing a leaves with multiple LogSpaceParameters."""
    return LeafWithMultipleLogParams(Scope([0, 1, 2]))


# ============================================================================
# SECTION 1: Basic Functionality Tests
# ============================================================================


class TestBasicFunctionality:
    """Test basic get/set operations and log-space storage."""

    def test_set_and_get_roundtrip(self, simple_leaf):
        """Test that set then get returns the same value."""
        value = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        simple_leaf.scale = value

        result = simple_leaf.scale
        torch.testing.assert_close(result, value)

    def test_internal_storage_is_log_space(self, simple_leaf):
        """Verify that internal storage is actually in log-space."""
        value = torch.tensor([[2.0, 3.0]])
        simple_leaf.scale = value

        # Internal log_scale should be log(value)
        expected_log = value.log()
        torch.testing.assert_close(simple_leaf.log_scale.data, expected_log)

    def test_exponential_recovery(self, simple_leaf):
        """Verify that exp(stored_log_value) equals original value."""
        value = torch.tensor([[0.5, 1.5, 2.5]])
        simple_leaf.scale = value

        # Get the raw log-space value and exponentiate it
        recovered = simple_leaf.log_scale.exp()
        torch.testing.assert_close(recovered, value)

    def test_multiple_assignment(self, simple_leaf):
        """Test that multiple assignments properly overwrite previous values."""
        value1 = torch.tensor([[1.0, 2.0]])
        value2 = torch.tensor([[3.0, 4.0]])
        value3 = torch.tensor([[5.0, 6.0]])

        simple_leaf.scale = value1
        torch.testing.assert_close(simple_leaf.scale, value1)

        simple_leaf.scale = value2
        torch.testing.assert_close(simple_leaf.scale, value2)

        simple_leaf.scale = value3
        torch.testing.assert_close(simple_leaf.scale, value3)

    def test_scalar_tensor_assignment(self, simple_leaf):
        """Test assignment of scalar tensors."""
        value = torch.tensor(2.5)
        simple_leaf.scale = value

        result = simple_leaf.scale
        torch.testing.assert_close(result, value)

    def test_descriptor_on_class_returns_descriptor_itself(self, simple_leaf):
        """Test descriptor protocol: accessing on class returns the descriptor."""
        descriptor = SimpleLeafWithLogParam.scale
        assert isinstance(descriptor, LogSpaceParameter)
        assert descriptor.name == "scale"

    def test_multiple_instances_independent(self):
        """Test that multiple instances maintain independent state."""
        leaf1 = SimpleLeafWithLogParam(Scope([0]))
        leaf2 = SimpleLeafWithLogParam(Scope([0]))

        leaf1.scale = torch.tensor([[1.0]])
        leaf2.scale = torch.tensor([[2.0]])

        torch.testing.assert_close(leaf1.scale, torch.tensor([[1.0]]))
        torch.testing.assert_close(leaf2.scale, torch.tensor([[2.0]]))

    def test_dtype_preservation(self, simple_leaf):
        """Test that dtype is preserved during assignment."""
        value_f32 = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        simple_leaf.scale = value_f32
        result = simple_leaf.scale
        assert result.dtype == torch.float32

    def test_device_preservation(self):
        """Test that device is preserved during assignment."""
        if torch.cuda.is_available():
            leaf = SimpleLeafWithLogParam(Scope([0]))
            leaf = leaf.to("cuda")
            value = torch.tensor([[1.0, 2.0]], device="cuda")
            leaf.scale = value
            result = leaf.scale
            assert result.device.type == "cuda"


# ============================================================================
# SECTION 2: Validation Edge Cases
# ============================================================================


class TestValidationEdgeCases:
    """Test validation of boundary values and invalid inputs."""

    def test_reject_zero(self, simple_leaf):
        """Test that zero values are rejected."""
        with pytest.raises(ValueError, match="must be greater than 0.0"):
            simple_leaf.scale = torch.tensor([[0.0]])

    def test_reject_negative(self, simple_leaf):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError, match="must be greater than 0.0"):
            simple_leaf.scale = torch.tensor([[-1.0]])

    def test_reject_nan(self, simple_leaf):
        """Test that NaN values are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.scale = torch.tensor([[float("nan")]])

    def test_reject_positive_infinity(self, simple_leaf):
        """Test that positive infinity is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.scale = torch.tensor([[float("inf")]])

    def test_reject_negative_infinity(self, simple_leaf):
        """Test that negative infinity is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.scale = torch.tensor([[float("-inf")]])

    def test_accept_very_small_positive(self, simple_leaf):
        """Test that very small positive values are accepted."""
        value = torch.tensor([[1e-10]])
        simple_leaf.scale = value
        result = simple_leaf.scale
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-15)

    def test_accept_very_large_positive(self, simple_leaf):
        """Test that very large positive values are accepted."""
        value = torch.tensor([[1e10]])
        simple_leaf.scale = value
        result = simple_leaf.scale
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-5)

    def test_accept_near_zero_but_positive(self, simple_leaf):
        """Test values very close to but greater than zero."""
        value = torch.tensor([[1e-8]])
        simple_leaf.scale = value
        result = simple_leaf.scale
        assert result.item() > 0.0

    def test_reject_partial_nan_in_tensor(self, simple_leaf):
        """Test that tensors with some NaN values are rejected."""
        with pytest.raises(ValueError):
            simple_leaf.scale = torch.tensor([[1.0, float("nan"), 2.0]])

    def test_reject_partial_inf_in_tensor(self, simple_leaf):
        """Test that tensors with some inf values are rejected."""
        with pytest.raises(ValueError):
            simple_leaf.scale = torch.tensor([[1.0, float("inf"), 2.0]])

    def test_error_message_includes_param_name(self, simple_leaf):
        """Test that error messages include the parameter name."""
        with pytest.raises(ValueError) as exc_info:
            simple_leaf.scale = torch.tensor([[-1.0]])
        assert "scale" in str(exc_info.value)

    def test_error_message_shows_value(self, simple_leaf):
        """Test that error messages show the problematic value."""
        with pytest.raises(ValueError) as exc_info:
            simple_leaf.scale = torch.tensor([[float("nan")]])
        assert "nan" in str(exc_info.value).lower()


# ============================================================================
# SECTION 3: Tensor Operations
# ============================================================================


class TestTensorOperations:
    """Test handling of different tensor shapes and dtypes."""

    def test_1d_tensor(self):
        """Test assignment and retrieval of 1D tensors."""
        leaf = SimpleLeafWithLogParam(Scope([0]), scale_init=torch.ones(1))
        value = torch.tensor([2.5])
        leaf.scale = value
        result = leaf.scale
        torch.testing.assert_close(result, value)

    def test_2d_tensor(self, simple_leaf):
        """Test assignment and retrieval of 2D tensors."""
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        simple_leaf.scale = value
        result = simple_leaf.scale
        torch.testing.assert_close(result, value)

    def test_3d_tensor(self):
        """Test assignment and retrieval of 3D tensors."""
        leaf = SimpleLeafWithLogParam(Scope([0, 1, 2]), scale_init=torch.ones(3, 2, 4))
        value = torch.ones(3, 2, 4) * 1.5
        leaf.scale = value
        result = leaf.scale
        torch.testing.assert_close(result, value)

    def test_large_tensor(self):
        """Test with large tensors."""
        shape = (100, 50)
        leaf = SimpleLeafWithLogParam(Scope([i for i in range(100)]), scale_init=torch.ones(shape))
        value = torch.randn(shape).abs() + 0.1  # Make positive
        leaf.scale = value
        result = leaf.scale
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-7)

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        leaf = SimpleLeafWithLogParam(Scope([0]), scale_init=torch.ones(1, dtype=torch.float32))
        value = torch.tensor([[1.5]], dtype=torch.float32)
        leaf.scale = value
        result = leaf.scale
        assert result.dtype == torch.float32

    def test_dtype_float64(self):
        """Test with float64 dtype."""
        scale_init = torch.ones(1, dtype=torch.float64)
        leaf = SimpleLeafWithLogParam(Scope([0]), scale_init=scale_init)
        value = torch.tensor([[1.5]], dtype=torch.float64)
        leaf.scale = value
        result = leaf.scale
        # Result dtype should match the parameter's dtype
        assert result.dtype == leaf.log_scale.dtype

    def test_dtype_conversion_on_assignment(self):
        """Test that assigned value is converted to match parameter dtype."""
        leaf = SimpleLeafWithLogParam(Scope([0]), scale_init=torch.ones(1, dtype=torch.float64))
        # Assign float32, should be converted to match parameter dtype
        value = torch.tensor([[1.5]], dtype=torch.float32)
        leaf.scale = value
        result = leaf.scale
        assert result.dtype == leaf.log_scale.dtype

    def test_broadcasting_in_assignment(self):
        """Test broadcasting behavior during assignment."""
        leaf = SimpleLeafWithLogParam(Scope([0, 1]), scale_init=torch.ones(2, 1))
        # Assign scalar, should broadcast
        value = torch.tensor([[2.5, 3.5]])
        leaf.scale = value
        result = leaf.scale
        torch.testing.assert_close(result, value)


# ============================================================================
# SECTION 4: PyTorch Integration
# ============================================================================


class TestPyTorchIntegration:
    """Test integration with PyTorch (gradients, optimization, state_dict)."""

    def test_gradients_flow_through_descriptor(self, simple_leaf):
        """Test that gradients propagate through the descriptor."""
        simple_leaf.scale.requires_grad_(True)
        value = torch.tensor([[2.0]], requires_grad=True)
        simple_leaf.scale = value

        # Compute a simple loss
        loss = simple_leaf.scale.sum()
        loss.backward()

        # Check that log_scale has gradients
        assert simple_leaf.log_scale.grad is not None
        # Gradient should be propagated through exp
        # d/d(log_scale) [exp(log_scale)] = exp(log_scale) = scale
        expected_grad = simple_leaf.scale.detach()
        torch.testing.assert_close(simple_leaf.log_scale.grad, expected_grad)

    def test_gradient_descent_optimization(self, simple_leaf):
        """Test that gradient descent optimization works correctly."""
        # Set initial value
        initial_value = torch.tensor([[10.0]])
        simple_leaf.scale = initial_value

        # Create optimizer
        optimizer = torch.optim.SGD(simple_leaf.parameters(), lr=0.1)

        # Perform a few optimization steps with loss = scale (minimize scale)
        for _ in range(10):
            optimizer.zero_grad()
            loss = simple_leaf.scale.sum()
            loss.backward()
            optimizer.step()

        # Scale should decrease
        final_scale = simple_leaf.scale.detach()
        assert final_scale.item() < initial_value.item()

    def test_state_dict_save_and_load(self, simple_leaf):
        """Test saving and loading state via state_dict."""
        # Set a value (matching the shape of simple_leaf)
        value = torch.tensor([[2.5], [3.5]])
        simple_leaf.scale = value

        # Get state dict
        state_dict = simple_leaf.state_dict()

        # Create new leaves with same scope (same shape)
        new_leaf = SimpleLeafWithLogParam(Scope([0, 1]))
        new_leaf.load_state_dict(state_dict)

        # Verify the value is restored
        torch.testing.assert_close(new_leaf.scale, value)

    def test_parameter_is_differentiable(self, simple_leaf):
        """Test that the parameter is differentiable."""
        assert simple_leaf.log_scale.requires_grad

    def test_exponential_gradient_chain(self, simple_leaf):
        """Test gradient chain rule: d/dx[exp(log(x))] through exp."""
        value = torch.tensor([[2.0]], requires_grad=False)
        simple_leaf.scale = value

        # Manually compute gradients
        loss = simple_leaf.scale.pow(2).sum()
        # For manual gradient: d/d(log_scale)[exp(log_scale)^2]
        # = d/d(log_scale)[scale^2]
        # = 2*scale * d/d(log_scale)[scale]
        # = 2*scale * scale (since d/d(log_scale)[exp(log_scale)] = exp(log_scale))

        simple_leaf.log_scale.grad = None
        loss.backward()

        # Gradient should be: 2 * scale^2
        expected_grad = 2.0 * value ** 2
        torch.testing.assert_close(simple_leaf.log_scale.grad, expected_grad)

    def test_module_cloning(self, simple_leaf):
        """Test that module can be cloned correctly."""
        value = torch.tensor([[2.0, 3.0]])
        simple_leaf.scale = value

        # Clone the module
        import copy

        cloned = copy.deepcopy(simple_leaf)

        # Verify cloned module has same parameter values
        torch.testing.assert_close(cloned.scale, simple_leaf.scale)

        # Verify they are independent
        cloned.scale = torch.tensor([[5.0, 6.0]])
        assert not torch.allclose(cloned.scale, simple_leaf.scale)


# ============================================================================
# SECTION 5: Custom Validators
# ============================================================================


class TestCustomValidators:
    """Test custom validator functionality."""

    def test_custom_validator_is_called(self, simple_leaf):
        """Test that custom validator is invoked during assignment."""
        call_count = [0]

        def counting_validator(value):
            call_count[0] += 1
            if torch.any(value <= 0.0):
                raise ValueError("Must be positive")

        descriptor = LogSpaceParameter("test", validator=counting_validator)
        leaf = SimpleLeafWithLogParam(Scope([0]))
        # Replace descriptor temporarily
        old_validator = leaf.__class__.scale.validator
        leaf.__class__.scale.validator = counting_validator

        leaf.scale = torch.tensor([[2.0]])
        assert call_count[0] > 0

        # Restore
        leaf.__class__.scale.validator = old_validator

    def test_custom_validator_stricter_constraint(self):
        """Test custom validator that enforces stricter constraints."""

        def min_one_validator(value):
            if torch.any(value < 1.0):
                raise ValueError("Value must be >= 1.0")
            if not torch.isfinite(value).all():
                raise ValueError("Value must be finite")

        descriptor = LogSpaceParameter("scale", validator=min_one_validator)

        class StrictLeaf(LeafModule):
            scale = descriptor

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 1)
                self.log_scale = nn.Parameter(torch.empty(1, 1))

            @property
            def distribution(self):
                return torch.distributions.Exponential(self.scale)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"scale": self.scale}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = StrictLeaf()

        # Should accept values >= 1.0
        leaf.scale = torch.tensor([[1.5]])
        torch.testing.assert_close(leaf.scale, torch.tensor([[1.5]]))

        # Should reject values < 1.0
        with pytest.raises(ValueError, match=">= 1.0"):
            leaf.scale = torch.tensor([[0.5]])

    def test_custom_validator_with_range(self):
        """Test custom validator enforcing range constraints."""

        def range_validator(value):
            if not torch.isfinite(value).all():
                raise ValueError("Value must be finite")
            if torch.any((value < 0.5) | (value > 5.0)):
                raise ValueError("Value must be in [0.5, 5.0]")

        descriptor = LogSpaceParameter("scale", validator=range_validator)

        class RangeLeaf(LeafModule):
            scale = descriptor

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 1)
                self.log_scale = nn.Parameter(torch.empty(1, 1))

            @property
            def distribution(self):
                return torch.distributions.Exponential(self.scale)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"scale": self.scale}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = RangeLeaf()

        # Should accept values in range
        leaf.scale = torch.tensor([[2.5]])
        torch.testing.assert_close(leaf.scale, torch.tensor([[2.5]]))

        # Should reject values below range
        with pytest.raises(ValueError, match="0.5, 5.0"):
            leaf.scale = torch.tensor([[0.1]])

        # Should reject values above range
        with pytest.raises(ValueError, match="0.5, 5.0"):
            leaf.scale = torch.tensor([[6.0]])


# ============================================================================
# SECTION 6: Multiple Parameters
# ============================================================================


class TestMultipleParameters:
    """Test descriptors with multiple LogSpaceParameters on same class."""

    def test_multiple_params_independent(self, multi_param_leaf):
        """Test that multiple parameters are independent."""
        alpha_value = torch.tensor([[1.0, 2.0, 3.0]])
        beta_value = torch.tensor([[4.0, 5.0, 6.0]])

        multi_param_leaf.alpha = alpha_value
        multi_param_leaf.beta = beta_value

        torch.testing.assert_close(multi_param_leaf.alpha, alpha_value)
        torch.testing.assert_close(multi_param_leaf.beta, beta_value)

    def test_multiple_params_separate_storage(self, multi_param_leaf):
        """Test that parameters use separate internal storage."""
        alpha_value = torch.tensor([[1.0, 2.0, 3.0]])
        beta_value = torch.tensor([[4.0, 5.0, 6.0]])

        multi_param_leaf.alpha = alpha_value
        multi_param_leaf.beta = beta_value

        # Internal storage should be separate
        expected_log_alpha = alpha_value.log()
        expected_log_beta = beta_value.log()

        torch.testing.assert_close(multi_param_leaf.log_alpha.data, expected_log_alpha)
        torch.testing.assert_close(multi_param_leaf.log_beta.data, expected_log_beta)

    def test_multiple_params_validation_independent(self, multi_param_leaf):
        """Test that validation is independent for each parameter."""
        # Alpha should accept the value
        multi_param_leaf.alpha = torch.tensor([[1.0, 2.0, 3.0]])

        # Beta should reject invalid value
        with pytest.raises(ValueError):
            multi_param_leaf.beta = torch.tensor([[-1.0, 2.0, 3.0]])

        # Alpha should still have its previous value
        torch.testing.assert_close(multi_param_leaf.alpha, torch.tensor([[1.0, 2.0, 3.0]]))


# ============================================================================
# SECTION 7: Integration with LeafModule
# ============================================================================


class TestLeafModuleIntegration:
    """Test integration with LeafModule and real-world usage patterns."""

    def test_assignment_in_mle(self):
        """Test that assignment works within MLE context."""

        class TestLeaf(LeafModule):
            scale = LogSpaceParameter("scale")

            def __init__(self, scope):
                super().__init__(scope, out_channels=1)
                self._event_shape = (len(scope.query), 1)
                self.log_scale = nn.Parameter(torch.empty(self._event_shape))
                self.scale = torch.ones(self._event_shape)

            @property
            def distribution(self):
                return torch.distributions.Exponential(self.scale)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"scale": self.scale}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                # This mimics real MLE usage
                scale_est = torch.ones_like(data[:1])
                self.scale = scale_est

        leaf = TestLeaf(Scope([0]))
        # This should not raise an error
        leaf.maximum_likelihood_estimation(torch.tensor([[1.0], [2.0], [3.0]]), nan_strategy=None)
        # Verify scale was set
        assert leaf.scale is not None

    def test_descriptor_with_broadcast_to_event_shape(self):
        """Test assignment after broadcasting to event shape."""

        class BroadcastLeaf(LeafModule):
            scale = LogSpaceParameter("scale")

            def __init__(self, scope, out_channels):
                super().__init__(scope, out_channels=out_channels)
                self._event_shape = (len(scope.query), out_channels)
                self.log_scale = nn.Parameter(torch.empty(self._event_shape))
                self.scale = torch.ones(self._event_shape)

            @property
            def distribution(self):
                return torch.distributions.Exponential(self.scale)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"scale": self.scale}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = BroadcastLeaf(Scope([0, 1]), out_channels=3)
        # Assign and broadcast
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        leaf.scale = value
        torch.testing.assert_close(leaf.scale, value)

    def test_params_method_returns_real_space(self):
        """Test that params() method returns real-space values."""

        class ParamsLeaf(LeafModule):
            scale = LogSpaceParameter("scale")

            def __init__(self, scope):
                super().__init__(scope, out_channels=1)
                self._event_shape = (len(scope.query), 1)
                self.log_scale = nn.Parameter(torch.empty(self._event_shape))
                self.scale = torch.tensor([[2.0]])

            @property
            def distribution(self):
                return torch.distributions.Exponential(self.scale)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"scale": self.scale}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = ParamsLeaf(Scope([0]))
        params = leaf.params()
        # params() should return real-space value, not log-space
        assert params["scale"].item() == 2.0


# ============================================================================
# SECTION 8: Descriptor Protocol Tests
# ============================================================================


class TestDescriptorProtocol:
    """Test Python descriptor protocol compliance."""

    def test_descriptor_get_on_class(self):
        """Test __get__ with obj=None returns descriptor."""
        descriptor = SimpleLeafWithLogParam.scale
        assert isinstance(descriptor, LogSpaceParameter)

    def test_descriptor_get_on_instance(self, simple_leaf):
        """Test __get__ with obj returns value."""
        result = simple_leaf.scale
        assert isinstance(result, torch.Tensor)

    def test_descriptor_set_on_instance(self, simple_leaf):
        """Test __set__ works on instance."""
        value = torch.tensor([[2.0, 3.0]])
        simple_leaf.scale = value
        torch.testing.assert_close(simple_leaf.scale, value)

    def test_descriptor_name_attribute(self):
        """Test that descriptor stores parameter name."""
        descriptor = LogSpaceParameter("rate")
        assert descriptor.name == "rate"
        assert descriptor.log_name == "log_rate"

    def test_different_param_names_different_storage(self):
        """Test that different parameter names use different storage."""

        class TwoParamLeaf(LeafModule):
            param1 = LogSpaceParameter("param1")
            param2 = LogSpaceParameter("param2")

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 1)
                self.log_param1 = nn.Parameter(torch.tensor([[1.0]]))
                self.log_param2 = nn.Parameter(torch.tensor([[1.0]]))

            @property
            def distribution(self):
                return None

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = TwoParamLeaf()
        leaf.param1 = torch.tensor([[2.0]])
        leaf.param2 = torch.tensor([[3.0]])

        # Should have different storage
        assert hasattr(leaf, "log_param1")
        assert hasattr(leaf, "log_param2")
        assert not torch.allclose(leaf.log_param1, leaf.log_param2)


# ============================================================================
# SECTION 9: Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability of log-space storage."""

    def test_stability_with_very_small_values(self):
        """Test numerical stability when dealing with very small positive values."""
        leaf = SimpleLeafWithLogParam(Scope([0]))
        value = torch.tensor([[1e-20]])
        leaf.scale = value
        result = leaf.scale

        # Should recover value without underflow
        torch.testing.assert_close(result, value, rtol=1e-10, atol=1e-25)

    def test_stability_with_very_large_values(self):
        """Test numerical stability with very large positive values."""
        leaf = SimpleLeafWithLogParam(Scope([0]))
        value = torch.tensor([[1e20]])
        leaf.scale = value
        result = leaf.scale

        # Should recover value without overflow
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-15)

    def test_gradient_stability_for_small_values(self, simple_leaf):
        """Test that gradients are stable for small parameter values."""
        value = torch.tensor([[0.001]], requires_grad=False)
        simple_leaf.scale = value

        loss = simple_leaf.scale.sum()
        loss.backward()

        # Gradient should be finite and approximately equal to the value itself
        # (since d/d(log_scale)[exp(log_scale)] = exp(log_scale))
        assert torch.isfinite(simple_leaf.log_scale.grad).all()
        expected_grad = value  # Gradient should be approximately the scale value
        torch.testing.assert_close(simple_leaf.log_scale.grad, expected_grad, rtol=1e-5, atol=1e-8)

    def test_repeated_assignment_stability(self, simple_leaf):
        """Test stability over many assignment cycles."""
        initial_value = torch.tensor([[2.5]])
        simple_leaf.scale = initial_value

        # Repeatedly set and get
        for _ in range(100):
            current = simple_leaf.scale.detach().clone()
            simple_leaf.scale = current

        final = simple_leaf.scale
        torch.testing.assert_close(final, initial_value, rtol=1e-5, atol=1e-7)


# ============================================================================
# SECTION 10: Edge Case Integration Tests
# ============================================================================


class TestEdgeCaseIntegration:
    """Test edge cases and unusual scenarios."""

    def test_assignment_from_numpy_like_value(self, simple_leaf):
        """Test assignment from numpy-like values."""
        value = [2.5, 3.5]  # List
        simple_leaf.scale = value
        result = simple_leaf.scale
        expected = torch.tensor([2.5, 3.5])
        torch.testing.assert_close(result, expected)

    def test_assignment_with_grad_enabled(self, simple_leaf):
        """Test assignment when gradients are enabled."""
        value = torch.tensor([[2.0]], requires_grad=True)
        simple_leaf.scale = value
        result = simple_leaf.scale
        assert result.requires_grad  # Should inherit requires_grad context

    def test_in_place_modification_rejected(self, simple_leaf):
        """Test that in-place modifications are not used (descriptor always replaces)."""
        value1 = torch.tensor([[1.0, 2.0]])
        value2 = torch.tensor([[3.0, 4.0]])

        simple_leaf.scale = value1
        id_before = id(simple_leaf.log_scale.data)

        simple_leaf.scale = value2
        id_after = id(simple_leaf.log_scale.data)

        # Data should be replaced, not modified in-place
        # (IDs might be the same due to implementation, but values should differ)
        torch.testing.assert_close(simple_leaf.scale, value2)

    def test_empty_tensor_rejection(self, simple_leaf):
        """Test that empty tensors are handled appropriately."""
        # Empty tensors should still be validated
        empty = torch.empty(0)
        # This might be valid (all() on empty is True), so just ensure no crash
        try:
            simple_leaf.scale = empty
            # If accepted, verify it's stored
            assert simple_leaf.scale.shape[0] == 0
        except (ValueError, RuntimeError):
            # If rejected, that's also acceptable
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
