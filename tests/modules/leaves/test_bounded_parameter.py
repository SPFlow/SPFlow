"""Comprehensive unit tests for BoundedParameter descriptor."""

import pytest
import torch
from torch import nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule, BoundedParameter


# ============================================================================
# Test Fixtures and Helper Classes
# ============================================================================


class SimpleLeafWithBoundedParam(LeafModule):
    """Simple test leaves module with BoundedParameter."""

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(self, scope, p_init=None, out_channels=1):
        super().__init__(scope, out_channels=out_channels)
        self._event_shape = (len(scope.query), out_channels)

        if p_init is None:
            p_init = torch.ones(self._event_shape) * 0.5
        else:
            p_init = torch.as_tensor(p_init)

        self.log_p = nn.Parameter(torch.empty_like(p_init))
        self.p = p_init.clone().detach()

    @property
    def distribution(self):
        return torch.distributions.Bernoulli(self.p)

    @property
    def _supported_value(self):
        return 0.0

    def params(self):
        return {"p": self.p}

    def _mle_compute_statistics(self, data, weights, bias_correction):
        pass


class LeafWithMultipleBoundedParams(LeafModule):
    """Test leaves with multiple BoundedParameters with different bounds."""

    p1 = BoundedParameter("p1", lb=0.0, ub=1.0)  # Probability
    alpha = BoundedParameter("alpha", lb=0.0, ub=None)  # Lower bounded only
    beta = BoundedParameter("beta", lb=None, ub=10.0)  # Upper bounded only

    def __init__(self, scope, out_channels=1):
        super().__init__(scope, out_channels=out_channels)
        self._event_shape = (len(scope.query), out_channels)

        p1_init = torch.ones(self._event_shape) * 0.5
        alpha_init = torch.ones(self._event_shape) * 2.0
        beta_init = torch.ones(self._event_shape) * 5.0

        self.log_p1 = nn.Parameter(torch.empty_like(p1_init))
        self.log_alpha = nn.Parameter(torch.empty_like(alpha_init))
        self.log_beta = nn.Parameter(torch.empty_like(beta_init))

        self.p1 = p1_init.clone().detach()
        self.alpha = alpha_init.clone().detach()
        self.beta = beta_init.clone().detach()

    @property
    def distribution(self):
        return torch.distributions.Bernoulli(self.p1)

    @property
    def _supported_value(self):
        return 0.0

    def params(self):
        return {"p1": self.p1, "alpha": self.alpha, "beta": self.beta}

    def _mle_compute_statistics(self, data, weights, bias_correction):
        pass


class LeafWithLowerBoundedParam(LeafModule):
    """Test leaves with lower-bounded parameter only."""

    x = BoundedParameter("x", lb=0.0, ub=None)

    def __init__(self, scope, out_channels=1):
        super().__init__(scope, out_channels=out_channels)
        self._event_shape = (len(scope.query), out_channels)

        x_init = torch.ones(self._event_shape)
        self.log_x = nn.Parameter(torch.empty_like(x_init))
        self.x = x_init.clone().detach()

    @property
    def distribution(self):
        return torch.distributions.Exponential(self.x)

    @property
    def _supported_value(self):
        return 0.0

    def params(self):
        return {"x": self.x}

    def _mle_compute_statistics(self, data, weights, bias_correction):
        pass


@pytest.fixture
def simple_leaf():
    """Fixture providing a simple leaves module with BoundedParameter."""
    return SimpleLeafWithBoundedParam(Scope([0, 1]))


@pytest.fixture
def multi_param_leaf():
    """Fixture providing a leaves with multiple BoundedParameters."""
    return LeafWithMultipleBoundedParams(Scope([0, 1, 2]))


@pytest.fixture
def lower_bounded_leaf():
    """Fixture providing a leaves with lower-bounded parameter only."""
    return LeafWithLowerBoundedParam(Scope([0, 1]))


# ============================================================================
# SECTION 1: Basic Functionality Tests
# ============================================================================


class TestBasicFunctionality:
    """Test basic get/set operations and bounded space storage."""

    def test_set_and_get_roundtrip(self, simple_leaf):
        """Test that set then get returns the same value."""
        value = torch.tensor([[0.3, 0.7], [0.5, 0.9]])
        simple_leaf.p = value

        result = simple_leaf.p
        torch.testing.assert_close(result, value)

    def test_internal_storage_is_unbounded(self, simple_leaf):
        """Verify that internal storage is in unbounded space via projection."""
        value = torch.tensor([[0.5, 0.8]])
        simple_leaf.p = value

        # Internal log_p should be proj_bounded_to_real(value, lb=0, ub=1)
        # which is log((value - 0) / (1 - value))
        expected_unbounded = torch.log((value - 0.0) / (1.0 - value))
        torch.testing.assert_close(simple_leaf.log_p.data, expected_unbounded)

    def test_projection_recovery(self, simple_leaf):
        """Verify that proj_real_to_bounded recovers original value."""
        value = torch.tensor([[0.2, 0.6, 0.9]])
        simple_leaf.p = value

        # Get the raw unbounded value and project it back
        recovered = torch.sigmoid(simple_leaf.log_p.data) * (1.0 - 0.0) + 0.0
        torch.testing.assert_close(recovered, value, rtol=1e-5, atol=1e-7)

    def test_multiple_assignment(self, simple_leaf):
        """Test that multiple assignments properly overwrite previous values."""
        value1 = torch.tensor([[0.2, 0.3]])
        value2 = torch.tensor([[0.5, 0.6]])
        value3 = torch.tensor([[0.8, 0.9]])

        simple_leaf.p = value1
        torch.testing.assert_close(simple_leaf.p, value1)

        simple_leaf.p = value2
        torch.testing.assert_close(simple_leaf.p, value2)

        simple_leaf.p = value3
        torch.testing.assert_close(simple_leaf.p, value3)

    def test_scalar_tensor_assignment(self, simple_leaf):
        """Test assignment of scalar tensors."""
        value = torch.tensor(0.7)
        simple_leaf.p = value

        result = simple_leaf.p
        torch.testing.assert_close(result, value)

    def test_descriptor_on_class_returns_descriptor_itself(self, simple_leaf):
        """Test descriptor protocol: accessing on class returns the descriptor."""
        descriptor = SimpleLeafWithBoundedParam.p
        assert isinstance(descriptor, BoundedParameter)
        assert descriptor.name == "p"
        assert descriptor.lb == 0.0
        assert descriptor.ub == 1.0

    def test_multiple_instances_independent(self):
        """Test that multiple instances maintain independent state."""
        leaf1 = SimpleLeafWithBoundedParam(Scope([0]))
        leaf2 = SimpleLeafWithBoundedParam(Scope([0]))

        leaf1.p = torch.tensor([[0.3]])
        leaf2.p = torch.tensor([[0.7]])

        torch.testing.assert_close(leaf1.p, torch.tensor([[0.3]]))
        torch.testing.assert_close(leaf2.p, torch.tensor([[0.7]]))

    def test_dtype_preservation(self, simple_leaf):
        """Test that dtype is preserved during assignment."""
        value_f32 = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
        simple_leaf.p = value_f32
        result = simple_leaf.p
        assert result.dtype == torch.float32

    def test_device_preservation(self):
        """Test that device is preserved during assignment."""
        if torch.cuda.is_available():
            leaf = SimpleLeafWithBoundedParam(Scope([0]))
            leaf = leaf.to("cuda")
            value = torch.tensor([[0.3, 0.7]], device="cuda")
            leaf.p = value
            result = leaf.p
            assert result.device.type == "cuda"


# ============================================================================
# SECTION 2: Validation Edge Cases
# ============================================================================


class TestValidationEdgeCases:
    """Test validation of boundary values and invalid inputs."""

    def test_reject_below_lower_bound(self, simple_leaf):
        """Test that values below lower bound are rejected."""
        with pytest.raises(ValueError, match="must be >= 0.0"):
            simple_leaf.p = torch.tensor([[-0.1]])

    def test_reject_above_upper_bound(self, simple_leaf):
        """Test that values above upper bound are rejected."""
        with pytest.raises(ValueError, match="must be <= 1.0"):
            simple_leaf.p = torch.tensor([[1.5]])

    def test_reject_nan(self, simple_leaf):
        """Test that NaN values are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.p = torch.tensor([[float("nan")]])

    def test_reject_positive_infinity(self, simple_leaf):
        """Test that positive infinity is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.p = torch.tensor([[float("inf")]])

    def test_reject_negative_infinity(self, simple_leaf):
        """Test that negative infinity is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.p = torch.tensor([[float("-inf")]])

    def test_accept_lower_bound_inclusive(self, simple_leaf):
        """Test that lower bound value is accepted (inclusive)."""
        value = torch.tensor([[0.0]])
        simple_leaf.p = value
        result = simple_leaf.p
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-8)

    def test_accept_upper_bound_inclusive(self, simple_leaf):
        """Test that upper bound value is accepted (inclusive)."""
        value = torch.tensor([[1.0]])
        simple_leaf.p = value
        result = simple_leaf.p
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-8)

    def test_accept_mid_range_value(self, simple_leaf):
        """Test that mid-range values are accepted."""
        value = torch.tensor([[0.5]])
        simple_leaf.p = value
        result = simple_leaf.p
        torch.testing.assert_close(result, value)

    def test_reject_partial_nan_in_tensor(self, simple_leaf):
        """Test that tensors with some NaN values are rejected."""
        with pytest.raises(ValueError):
            simple_leaf.p = torch.tensor([[0.3, float("nan"), 0.7]])

    def test_reject_partial_out_of_bounds_in_tensor(self, simple_leaf):
        """Test that tensors with some out-of-bounds values are rejected."""
        with pytest.raises(ValueError):
            simple_leaf.p = torch.tensor([[0.3, 1.5, 0.7]])

    def test_error_message_includes_param_name(self, simple_leaf):
        """Test that error messages include the parameter name."""
        with pytest.raises(ValueError) as exc_info:
            simple_leaf.p = torch.tensor([[-0.1]])
        assert "p" in str(exc_info.value)

    def test_lower_bounded_only_rejects_below(self, multi_param_leaf):
        """Test lower-bounded parameter rejects values below bound."""
        with pytest.raises(ValueError):
            multi_param_leaf.alpha = torch.tensor([[-1.0, 2.0, 3.0]])

    def test_lower_bounded_only_accepts_above(self, multi_param_leaf):
        """Test lower-bounded parameter accepts values above bound."""
        value = torch.tensor([[0.1, 2.0, 3.0]])
        multi_param_leaf.alpha = value
        result = multi_param_leaf.alpha
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-7)

    def test_lower_bounded_only_accepts_large_values(self, multi_param_leaf):
        """Test lower-bounded parameter accepts large values."""
        value = torch.tensor([[1e10]])
        multi_param_leaf.alpha = value
        result = multi_param_leaf.alpha
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-5)

    def test_upper_bounded_only_rejects_above(self, multi_param_leaf):
        """Test upper-bounded parameter rejects values above bound."""
        with pytest.raises(ValueError):
            multi_param_leaf.beta = torch.tensor([[11.0, 5.0, 3.0]])

    def test_upper_bounded_only_accepts_below(self, multi_param_leaf):
        """Test upper-bounded parameter accepts values below bound."""
        value = torch.tensor([[5.0, 3.0, 1.0]])
        multi_param_leaf.beta = value
        result = multi_param_leaf.beta
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-7)

    def test_upper_bounded_only_accepts_negative_values(self, multi_param_leaf):
        """Test upper-bounded parameter accepts negative values."""
        value = torch.tensor([[-1e10]])
        multi_param_leaf.beta = value
        result = multi_param_leaf.beta
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-5)

    def test_lower_bounded_accepts_positive_values(self, lower_bounded_leaf):
        """Test lower-bounded parameter accepts positive finite values."""
        value = torch.tensor([[0.01, 1.0, 10.0, 1e5]])
        lower_bounded_leaf.x = value
        result = lower_bounded_leaf.x
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-5)


# ============================================================================
# SECTION 3: Tensor Operations
# ============================================================================


class TestTensorOperations:
    """Test handling of different tensor shapes and dtypes."""

    def test_1d_tensor(self):
        """Test assignment and retrieval of 1D tensors."""
        leaf = SimpleLeafWithBoundedParam(Scope([0]), p_init=torch.ones(1) * 0.5)
        value = torch.tensor([0.7])
        leaf.p = value
        result = leaf.p
        torch.testing.assert_close(result, value)

    def test_2d_tensor(self, simple_leaf):
        """Test assignment and retrieval of 2D tensors."""
        value = torch.tensor([[0.2, 0.3], [0.7, 0.8]])
        simple_leaf.p = value
        result = simple_leaf.p
        torch.testing.assert_close(result, value)

    def test_3d_tensor(self):
        """Test assignment and retrieval of 3D tensors."""
        leaf = SimpleLeafWithBoundedParam(Scope([0, 1, 2]), p_init=torch.ones(3, 2, 4) * 0.5)
        value = torch.ones(3, 2, 4) * 0.3
        leaf.p = value
        result = leaf.p
        torch.testing.assert_close(result, value)

    def test_large_tensor(self):
        """Test with large tensors."""
        shape = (100, 50)
        leaf = SimpleLeafWithBoundedParam(Scope([i for i in range(100)]), p_init=torch.ones(shape) * 0.5)
        value = torch.rand(shape) * 0.8 + 0.1  # Values in [0.1, 0.9]
        leaf.p = value
        result = leaf.p
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-7)

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        leaf = SimpleLeafWithBoundedParam(Scope([0]), p_init=torch.ones(1, dtype=torch.float32) * 0.5)
        value = torch.tensor([[0.7]], dtype=torch.float32)
        leaf.p = value
        result = leaf.p
        assert result.dtype == torch.float32

    def test_dtype_float64(self):
        """Test with float64 dtype."""
        p_init = torch.ones(1, dtype=torch.float64) * 0.5
        leaf = SimpleLeafWithBoundedParam(Scope([0]), p_init=p_init)
        value = torch.tensor([[0.7]], dtype=torch.float64)
        leaf.p = value
        result = leaf.p
        # Result dtype should match the parameter's dtype
        assert result.dtype == leaf.log_p.dtype

    def test_dtype_conversion_on_assignment(self):
        """Test that assigned value is converted to match parameter dtype."""
        leaf = SimpleLeafWithBoundedParam(Scope([0]), p_init=torch.ones(1, dtype=torch.float64) * 0.5)
        # Assign float32, should be converted to match parameter dtype
        value = torch.tensor([[0.7]], dtype=torch.float32)
        leaf.p = value
        result = leaf.p
        assert result.dtype == leaf.log_p.dtype

    def test_broadcasting_in_assignment(self):
        """Test broadcasting behavior during assignment."""
        leaf = SimpleLeafWithBoundedParam(Scope([0, 1]), p_init=torch.ones(2, 1) * 0.5)
        # Assign scalar, should broadcast
        value = torch.tensor([[0.3, 0.7]])
        leaf.p = value
        result = leaf.p
        torch.testing.assert_close(result, value)


# ============================================================================
# SECTION 4: PyTorch Integration
# ============================================================================


class TestPyTorchIntegration:
    """Test integration with PyTorch (gradients, optimization, state_dict)."""

    def test_gradients_flow_through_descriptor(self, simple_leaf):
        """Test that gradients propagate through the descriptor."""
        value = torch.tensor([[0.5]], requires_grad=False)
        simple_leaf.p = value

        # Compute a simple loss
        loss = simple_leaf.p.sum()
        loss.backward()

        # Check that log_p has gradients
        assert simple_leaf.log_p.grad is not None
        # Gradient should be propagated through sigmoid projection
        assert torch.isfinite(simple_leaf.log_p.grad).all()

    def test_gradient_descent_optimization(self, simple_leaf):
        """Test that gradient descent optimization works correctly."""
        # Set initial value
        initial_value = torch.tensor([[0.9]])
        simple_leaf.p = initial_value

        # Create optimizer
        optimizer = torch.optim.SGD(simple_leaf.parameters(), lr=0.1)

        # Perform optimization steps with loss = (p - 0.1)^2 (minimize to 0.1)
        target = 0.1
        for _ in range(10):
            optimizer.zero_grad()
            loss = (simple_leaf.p - target) ** 2
            loss.backward()
            optimizer.step()

        # p should move towards target
        final_p = simple_leaf.p.detach()
        assert final_p.item() < initial_value.item()

    def test_state_dict_save_and_load(self, simple_leaf):
        """Test saving and loading state via state_dict."""
        # Set a value
        value = torch.tensor([[0.3], [0.7]])
        simple_leaf.p = value

        # Get state dict
        state_dict = simple_leaf.state_dict()

        # Create new leaves with same scope
        new_leaf = SimpleLeafWithBoundedParam(Scope([0, 1]))
        new_leaf.load_state_dict(state_dict)

        # Verify the value is restored
        torch.testing.assert_close(new_leaf.p, value, rtol=1e-5, atol=1e-7)

    def test_parameter_is_differentiable(self, simple_leaf):
        """Test that the parameter is differentiable."""
        assert simple_leaf.log_p.requires_grad

    def test_module_cloning(self, simple_leaf):
        """Test that module can be cloned correctly."""
        value = torch.tensor([[0.3, 0.7]])
        simple_leaf.p = value

        # Clone the module
        import copy

        cloned = copy.deepcopy(simple_leaf)

        # Verify cloned module has same parameter values
        torch.testing.assert_close(cloned.p, simple_leaf.p, rtol=1e-5, atol=1e-7)

        # Verify they are independent
        cloned.p = torch.tensor([[0.5, 0.9]])
        assert not torch.allclose(cloned.p, simple_leaf.p)

    def test_gradient_chain_sigmoid_projection(self, simple_leaf):
        """Test gradient chain rule through sigmoid projection."""
        value = torch.tensor([[0.5]], requires_grad=False)
        simple_leaf.p = value

        # Compute loss
        loss = simple_leaf.p.pow(2).sum()
        loss.backward()

        # Check that gradient is finite
        assert torch.isfinite(simple_leaf.log_p.grad).all()
        # Gradient should be non-zero
        assert not torch.allclose(simple_leaf.log_p.grad, torch.zeros_like(simple_leaf.log_p.grad))


# ============================================================================
# SECTION 5: Custom Validators
# ============================================================================


class TestCustomValidators:
    """Test custom validator functionality."""

    def test_custom_validator_is_called(self):
        """Test that custom validator is invoked during assignment."""
        call_count = [0]

        def counting_validator(value):
            call_count[0] += 1
            if torch.any(value < 0.2) or torch.any(value > 0.8):
                raise ValueError("Must be in [0.2, 0.8]")

        descriptor = BoundedParameter("p", lb=0.0, ub=1.0, validator=counting_validator)

        class CustomLeaf(LeafModule):
            p = descriptor

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 1)
                self.log_p = nn.Parameter(torch.empty(1, 1))

            @property
            def distribution(self):
                return torch.distributions.Bernoulli(self.p)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"p": self.p}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = CustomLeaf()
        leaf.p = torch.tensor([[0.5]])
        assert call_count[0] > 0

    def test_custom_validator_stricter_constraint(self):
        """Test custom validator that enforces stricter constraints."""

        def range_validator(value):
            if not torch.isfinite(value).all():
                raise ValueError("Value must be finite")
            if torch.any((value < 0.2) | (value > 0.8)):
                raise ValueError("Value must be in [0.2, 0.8]")

        descriptor = BoundedParameter("p", lb=0.0, ub=1.0, validator=range_validator)

        class StrictLeaf(LeafModule):
            p = descriptor

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 1)
                self.log_p = nn.Parameter(torch.empty(1, 1))

            @property
            def distribution(self):
                return torch.distributions.Bernoulli(self.p)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"p": self.p}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = StrictLeaf()

        # Should accept values in [0.2, 0.8]
        leaf.p = torch.tensor([[0.5]])
        torch.testing.assert_close(leaf.p, torch.tensor([[0.5]]))

        # Should reject values below 0.2
        with pytest.raises(ValueError, match="0.2, 0.8"):
            leaf.p = torch.tensor([[0.1]])

        # Should reject values above 0.8
        with pytest.raises(ValueError, match="0.2, 0.8"):
            leaf.p = torch.tensor([[0.9]])


# ============================================================================
# SECTION 6: Multiple Parameters
# ============================================================================


class TestMultipleParameters:
    """Test descriptors with multiple BoundedParameters on same class."""

    def test_multiple_params_with_different_bounds(self, multi_param_leaf):
        """Test that multiple parameters with different bounds work independently."""
        p1_value = torch.tensor([[0.3, 0.7, 0.5]])
        alpha_value = torch.tensor([[1.0, 2.0, 3.0]])
        beta_value = torch.tensor([[4.0, 5.0, 6.0]])

        multi_param_leaf.p1 = p1_value
        multi_param_leaf.alpha = alpha_value
        multi_param_leaf.beta = beta_value

        torch.testing.assert_close(multi_param_leaf.p1, p1_value, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(multi_param_leaf.alpha, alpha_value, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(multi_param_leaf.beta, beta_value, rtol=1e-5, atol=1e-7)

    def test_multiple_params_separate_storage(self, multi_param_leaf):
        """Test that parameters use separate internal storage."""
        p1_value = torch.tensor([[0.5, 0.6, 0.7]])
        alpha_value = torch.tensor([[1.0, 2.0, 3.0]])
        beta_value = torch.tensor([[4.0, 5.0, 6.0]])

        multi_param_leaf.p1 = p1_value
        multi_param_leaf.alpha = alpha_value
        multi_param_leaf.beta = beta_value

        # Internal storage should be separate
        assert not torch.allclose(multi_param_leaf.log_p1, multi_param_leaf.log_alpha)
        assert not torch.allclose(multi_param_leaf.log_p1, multi_param_leaf.log_beta)
        assert not torch.allclose(multi_param_leaf.log_alpha, multi_param_leaf.log_beta)

    def test_multiple_params_validation_independent(self, multi_param_leaf):
        """Test that validation is independent for each parameter."""
        # p1 should accept the value
        multi_param_leaf.p1 = torch.tensor([[0.3, 0.7, 0.5]])

        # alpha should reject values below 0 (but we're giving 1+)
        multi_param_leaf.alpha = torch.tensor([[1.0, 2.0, 3.0]])

        # beta should reject values above 10
        with pytest.raises(ValueError):
            multi_param_leaf.beta = torch.tensor([[15.0, 5.0, 3.0]])

        # p1 and alpha should still have their values
        torch.testing.assert_close(multi_param_leaf.p1, torch.tensor([[0.3, 0.7, 0.5]]), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(multi_param_leaf.alpha, torch.tensor([[1.0, 2.0, 3.0]]), rtol=1e-5, atol=1e-7)


# ============================================================================
# SECTION 7: Integration with LeafModule
# ============================================================================


class TestLeafModuleIntegration:
    """Test integration with LeafModule and real-world usage patterns."""

    def test_assignment_in_mle(self):
        """Test that assignment works within MLE context."""

        class TestLeaf(LeafModule):
            p = BoundedParameter("p", lb=0.0, ub=1.0)

            def __init__(self, scope):
                super().__init__(scope, out_channels=1)
                self._event_shape = (len(scope.query), 1)
                self.log_p = nn.Parameter(torch.empty(self._event_shape))
                self.p = torch.ones(self._event_shape) * 0.5

            @property
            def distribution(self):
                return torch.distributions.Bernoulli(self.p)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"p": self.p}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                # This mimics real MLE usage
                p_est = torch.ones_like(data[:1]) * 0.5
                self.p = p_est

        leaf = TestLeaf(Scope([0]))
        # This should not raise an error
        leaf.maximum_likelihood_estimation(
            torch.tensor([[1.0], [0.0], [1.0]]), nan_strategy=None
        )
        # Verify p was set
        assert leaf.p is not None
        # Verify p is in valid range
        assert (leaf.p >= 0.0).all() and (leaf.p <= 1.0).all()

    def test_descriptor_with_broadcast_to_event_shape(self):
        """Test assignment after broadcasting to event shape."""

        class BroadcastLeaf(LeafModule):
            p = BoundedParameter("p", lb=0.0, ub=1.0)

            def __init__(self, scope, out_channels):
                super().__init__(scope, out_channels=out_channels)
                self._event_shape = (len(scope.query), out_channels)
                self.log_p = nn.Parameter(torch.empty(self._event_shape))
                self.p = torch.ones(self._event_shape) * 0.5

            @property
            def distribution(self):
                return torch.distributions.Bernoulli(self.p)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"p": self.p}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = BroadcastLeaf(Scope([0, 1]), out_channels=3)
        # Assign and broadcast
        value = torch.tensor([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]])
        leaf.p = value
        torch.testing.assert_close(leaf.p, value, rtol=1e-5, atol=1e-7)

    def test_params_method_returns_bounded_space(self):
        """Test that params() method returns bounded-space values."""

        class ParamsLeaf(LeafModule):
            p = BoundedParameter("p", lb=0.0, ub=1.0)

            def __init__(self, scope):
                super().__init__(scope, out_channels=1)
                self._event_shape = (len(scope.query), 1)
                self.log_p = nn.Parameter(torch.empty(self._event_shape))
                self.p = torch.tensor([[0.5]])

            @property
            def distribution(self):
                return torch.distributions.Bernoulli(self.p)

            @property
            def _supported_value(self):
                return 0.0

            def params(self):
                return {"p": self.p}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = ParamsLeaf(Scope([0]))
        params = leaf.params()
        # params() should return bounded-space value
        assert (params["p"] >= 0.0).all() and (params["p"] <= 1.0).all()
        torch.testing.assert_close(params["p"], torch.tensor([[0.5]]), rtol=1e-5, atol=1e-7)


# ============================================================================
# SECTION 8: Descriptor Protocol Tests
# ============================================================================


class TestDescriptorProtocol:
    """Test Python descriptor protocol compliance."""

    def test_descriptor_get_on_class(self):
        """Test __get__ with obj=None returns descriptor."""
        descriptor = SimpleLeafWithBoundedParam.p
        assert isinstance(descriptor, BoundedParameter)

    def test_descriptor_get_on_instance(self, simple_leaf):
        """Test __get__ with obj returns value."""
        result = simple_leaf.p
        assert isinstance(result, torch.Tensor)

    def test_descriptor_set_on_instance(self, simple_leaf):
        """Test __set__ works on instance."""
        value = torch.tensor([[0.3, 0.7]])
        simple_leaf.p = value
        torch.testing.assert_close(simple_leaf.p, value)

    def test_descriptor_name_attribute(self):
        """Test that descriptor stores parameter name."""
        descriptor = BoundedParameter("probability", lb=0.0, ub=1.0)
        assert descriptor.name == "probability"
        assert descriptor.log_name == "log_probability"

    def test_different_param_names_different_storage(self):
        """Test that different parameter names use different storage."""

        class TwoParamLeaf(LeafModule):
            param1 = BoundedParameter("param1", lb=0.0, ub=1.0)
            param2 = BoundedParameter("param2", lb=0.0, ub=10.0)

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 1)
                # Initialize with different unbounded values
                self.log_param1 = nn.Parameter(torch.tensor([[-1.0]]))
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
        leaf.param1 = torch.tensor([[0.1]])  # With bounds [0, 1]
        leaf.param2 = torch.tensor([[9.0]])  # With bounds [0, 10]

        # Should have different storage attributes
        assert hasattr(leaf, "log_param1")
        assert hasattr(leaf, "log_param2")
        # They should be different Parameter objects
        assert leaf.log_param1 is not leaf.log_param2
        # After assignment with very different values in their bounds, they should have very different stored values
        # param1=0.1 in [0,1] -> log(0.1/0.9) ≈ -2.197
        # param2=9.0 in [0,10] -> log(9/1) ≈ 2.197
        # These should be very different (sum of absolute values should be large)
        assert torch.abs(leaf.log_param1) + torch.abs(leaf.log_param2) > 3.0

    def test_bounds_stored_in_descriptor(self):
        """Test that bounds are stored in the descriptor."""
        descriptor = BoundedParameter("x", lb=0.5, ub=2.5)
        assert descriptor.lb == 0.5
        assert descriptor.ub == 2.5

    def test_bounds_as_tensors(self):
        """Test that bounds can be specified as tensors."""
        lb = torch.tensor(0.0)
        ub = torch.tensor(1.0)
        descriptor = BoundedParameter("p", lb=lb, ub=ub)
        assert descriptor.lb == lb
        assert descriptor.ub == ub


# ============================================================================
# SECTION 9: Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability of bounded parameter projections."""

    def test_stability_near_lower_bound(self, simple_leaf):
        """Test numerical stability when value is very close to lower bound."""
        value = torch.tensor([[1e-7]])
        simple_leaf.p = value
        result = simple_leaf.p

        # Should recover value with reasonable accuracy
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-8)

    def test_stability_near_upper_bound(self, simple_leaf):
        """Test numerical stability when value is very close to upper bound."""
        value = torch.tensor([[1.0 - 1e-7]])
        simple_leaf.p = value
        result = simple_leaf.p

        # Should recover value with reasonable accuracy
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-8)

    def test_stability_repeated_assignment(self, simple_leaf):
        """Test stability over many assignment cycles."""
        initial_value = torch.tensor([[0.5]])
        simple_leaf.p = initial_value

        # Repeatedly set and get
        for _ in range(100):
            current = simple_leaf.p.detach().clone()
            simple_leaf.p = current

        final = simple_leaf.p
        torch.testing.assert_close(final, initial_value, rtol=1e-5, atol=1e-7)

    def test_gradient_stability(self, simple_leaf):
        """Test that gradients remain stable through projection."""
        value = torch.tensor([[0.5]], requires_grad=False)
        simple_leaf.p = value

        loss = simple_leaf.p.sum()
        loss.backward()

        # Gradient should be finite
        assert torch.isfinite(simple_leaf.log_p.grad).all()
        # Gradient should be non-zero
        assert (simple_leaf.log_p.grad.abs() > 0).all()

    def test_lower_bounded_stability_large_value(self, multi_param_leaf):
        """Test numerical stability with lower-bounded large values."""
        value = torch.tensor([[1e10]])
        multi_param_leaf.alpha = value
        result = multi_param_leaf.alpha

        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-5)

    def test_upper_bounded_stability_negative_value(self, multi_param_leaf):
        """Test numerical stability with upper-bounded negative values."""
        value = torch.tensor([[-1e10]])
        multi_param_leaf.beta = value
        result = multi_param_leaf.beta

        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-5)


# ============================================================================
# SECTION 10: Edge Case Integration Tests
# ============================================================================


class TestEdgeCaseIntegration:
    """Test edge cases and unusual scenarios."""

    def test_assignment_from_numpy_like_value(self, simple_leaf):
        """Test assignment from numpy-like values."""
        value = [0.3, 0.7]  # List
        simple_leaf.p = value
        result = simple_leaf.p
        expected = torch.tensor([0.3, 0.7])
        torch.testing.assert_close(result, expected)

    def test_assignment_with_grad_enabled(self, simple_leaf):
        """Test assignment when gradients are enabled."""
        value = torch.tensor([[0.5]], requires_grad=True)
        simple_leaf.p = value
        result = simple_leaf.p
        assert result.requires_grad

    def test_boundary_values_roundtrip(self, simple_leaf):
        """Test that boundary values survive roundtrip without overflow."""
        for boundary_value in [0.0, 1.0]:
            value = torch.tensor([[boundary_value]])
            simple_leaf.p = value
            result = simple_leaf.p
            torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-8)

    def test_multiple_gradient_steps(self, simple_leaf):
        """Test that gradients flow correctly over multiple steps."""
        value = torch.tensor([[0.5]])
        simple_leaf.p = value

        optimizer = torch.optim.SGD(simple_leaf.parameters(), lr=0.05)

        for step in range(5):
            optimizer.zero_grad()
            loss = (simple_leaf.p - 0.3) ** 2
            loss.backward()

            # Gradient should be finite at each step
            assert torch.isfinite(simple_leaf.log_p.grad).all()

            optimizer.step()

            # p should remain in valid bounds
            assert (simple_leaf.p >= 0.0).all() and (simple_leaf.p <= 1.0).all()

    def test_lower_bounded_param_large_range(self, lower_bounded_leaf):
        """Test lower-bounded parameter accepts large range of positive values."""
        values = [
            torch.tensor([[1e-8]]),
            torch.tensor([[1e-5]]),
            torch.tensor([[0.001]]),
            torch.tensor([[1.0]]),
            torch.tensor([[1000.0]]),
            torch.tensor([[1e5]]),
            torch.tensor([[1e10]]),
        ]

        for value in values:
            lower_bounded_leaf.x = value
            result = lower_bounded_leaf.x
            torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
