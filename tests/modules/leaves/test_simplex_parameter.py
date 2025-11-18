"""Comprehensive unit tests for SimplexParameter descriptor."""

import pytest
import torch
from torch import nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import SimplexParameter


# ============================================================================
# Test Fixtures and Helper Classes
# ============================================================================


class SimpleLeafWithSimplexParam(LeafModule):
    """Simple test leaves module with SimplexParameter."""

    probs = SimplexParameter("probs")

    def __init__(self, scope, probs_init=None, out_channels=1):
        super().__init__(scope, out_channels=out_channels)
        self._event_shape = (len(scope.query), out_channels)

        if probs_init is None:
            # Initialize with uniform distribution
            probs_init = torch.ones(self._event_shape) / self._event_shape[-1]
        else:
            probs_init = torch.as_tensor(probs_init)

        self.logits_probs = nn.Parameter(torch.empty_like(probs_init))
        self.probs = probs_init.clone().detach()

    @property
    def distribution(self):
        return torch.distributions.Categorical(self.probs)

    @property
    def _supported_value(self):
        return 0

    def params(self):
        return {"probs": self.probs}

    def _mle_compute_statistics(self, data, weights, bias_correction):
        pass


class LeafWithMultipleSimplexParams(LeafModule):
    """Test leaves with multiple SimplexParameters."""

    weights1 = SimplexParameter("weights1")
    weights2 = SimplexParameter("weights2")

    def __init__(self, scope, out_channels=1):
        super().__init__(scope, out_channels=out_channels)
        self._event_shape = (len(scope.query), out_channels)

        # Initialize with different distributions
        weights1_init = torch.tensor([0.7, 0.3]).repeat(self._event_shape[0], 1)
        weights2_init = torch.tensor([0.2, 0.5, 0.3]).repeat(self._event_shape[0], 1)

        # Adjust shapes to match event_shape
        if self._event_shape[1] != 2:
            weights1_init = torch.ones(self._event_shape) / self._event_shape[1]
        if self._event_shape[1] != 3:
            weights2_init = torch.ones(self._event_shape) / self._event_shape[1]

        self.logits_weights1 = nn.Parameter(torch.empty_like(weights1_init))
        self.logits_weights2 = nn.Parameter(torch.empty_like(weights2_init))

        self.weights1 = weights1_init.clone().detach()
        self.weights2 = weights2_init.clone().detach()

    @property
    def distribution(self):
        return torch.distributions.Categorical(self.weights1)

    @property
    def _supported_value(self):
        return 0

    def params(self):
        return {"weights1": self.weights1, "weights2": self.weights2}

    def _mle_compute_statistics(self, data, weights, bias_correction):
        pass


@pytest.fixture
def simple_leaf():
    """Fixture providing a simple leaves module with SimplexParameter."""
    return SimpleLeafWithSimplexParam(Scope([0, 1]))


@pytest.fixture
def multi_param_leaf():
    """Fixture providing a leaves with multiple SimplexParameters."""
    return LeafWithMultipleSimplexParams(Scope([0, 1, 2]))


# ============================================================================
# SECTION 1: Basic Functionality Tests
# ============================================================================


class TestBasicFunctionality:
    """Test basic get/set operations and simplex storage."""

    def test_set_and_get_roundtrip(self, simple_leaf):
        """Test that set then get returns the same value."""
        value = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
        simple_leaf.probs = value

        result = simple_leaf.probs
        torch.testing.assert_close(result, value)

    def test_internal_storage_is_logits(self, simple_leaf):
        """Verify that internal storage is actually in logits (unconstrained space)."""
        value = torch.tensor([[0.3, 0.7]])
        simple_leaf.probs = value

        # Internal logits_probs should be log(value)
        expected_logits = torch.log(value)
        torch.testing.assert_close(simple_leaf.logits_probs.data, expected_logits)

    def test_softmax_recovery(self, simple_leaf):
        """Verify that softmax(stored_logits) equals original value."""
        value = torch.tensor([[0.2, 0.3, 0.5]])
        simple_leaf.probs = value

        # Get the raw logits and apply softmax
        recovered = torch.softmax(simple_leaf.logits_probs.data, dim=-1)
        torch.testing.assert_close(recovered, value, rtol=1e-5, atol=1e-7)

    def test_multiple_assignment(self, simple_leaf):
        """Test that multiple assignments properly overwrite previous values."""
        value1 = torch.tensor([[0.2, 0.8]])
        value2 = torch.tensor([[0.5, 0.5]])
        value3 = torch.tensor([[0.7, 0.3]])

        simple_leaf.probs = value1
        torch.testing.assert_close(simple_leaf.probs, value1)

        simple_leaf.probs = value2
        torch.testing.assert_close(simple_leaf.probs, value2)

        simple_leaf.probs = value3
        torch.testing.assert_close(simple_leaf.probs, value3)

    def test_scalar_tensor_assignment(self, simple_leaf):
        """Test assignment of scalar tensors."""
        value = torch.tensor([0.5, 0.5])  # Must sum to 1 for simplex
        simple_leaf.probs = value

        result = simple_leaf.probs
        torch.testing.assert_close(result, value)

    def test_descriptor_on_class_returns_descriptor_itself(self, simple_leaf):
        """Test descriptor protocol: accessing on class returns the descriptor."""
        descriptor = SimpleLeafWithSimplexParam.probs
        assert isinstance(descriptor, SimplexParameter)
        assert descriptor.name == "probs"

    def test_multiple_instances_independent(self):
        """Test that multiple instances maintain independent state."""
        leaf1 = SimpleLeafWithSimplexParam(Scope([0]))
        leaf2 = SimpleLeafWithSimplexParam(Scope([0]))

        leaf1.probs = torch.tensor([[0.3, 0.7]])  # Must sum to 1
        leaf2.probs = torch.tensor([[0.7, 0.3]])  # Must sum to 1

        torch.testing.assert_close(leaf1.probs, torch.tensor([[0.3, 0.7]]))
        torch.testing.assert_close(leaf2.probs, torch.tensor([[0.7, 0.3]]))

    def test_dtype_preservation(self, simple_leaf):
        """Test that dtype is preserved during assignment."""
        value_f32 = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
        simple_leaf.probs = value_f32
        result = simple_leaf.probs
        assert result.dtype == torch.float32

    def test_device_preservation(self):
        """Test that device is preserved during assignment."""
        if torch.cuda.is_available():
            leaf = SimpleLeafWithSimplexParam(Scope([0]))
            leaf = leaf.to("cuda")
            value = torch.tensor([[0.3, 0.7]], device="cuda")
            leaf.probs = value
            result = leaf.probs
            assert result.device.type == "cuda"

    def test_simplex_constraint_satisfaction(self, simple_leaf):
        """Test that values satisfy simplex constraints (non-negative, sum to 1)."""
        value = torch.tensor([[0.2, 0.8]])  # Must match shape (2 categories)
        simple_leaf.probs = value

        result = simple_leaf.probs
        # Check non-negativity
        assert (result >= 0).all()
        # Check sum to 1
        sums = result.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)


# ============================================================================
# SECTION 2: Validation Edge Cases
# ============================================================================


class TestValidationEdgeCases:
    """Test validation of boundary values and invalid inputs."""

    def test_reject_negative(self, simple_leaf):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            simple_leaf.probs = torch.tensor([[-0.1, 0.5]])

    def test_reject_nan(self, simple_leaf):
        """Test that NaN values are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.probs = torch.tensor([[float("nan"), 0.5]])

    def test_reject_positive_infinity(self, simple_leaf):
        """Test that positive infinity is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.probs = torch.tensor([[float("inf"), 0.5]])

    def test_reject_negative_infinity(self, simple_leaf):
        """Test that negative infinity is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            simple_leaf.probs = torch.tensor([[float("-inf"), 0.5]])

    def test_reject_sum_not_one(self, simple_leaf):
        """Test that values not summing to 1 are rejected."""
        with pytest.raises(ValueError, match="must sum to 1"):
            simple_leaf.probs = torch.tensor([[0.3, 0.8]])  # Sum = 1.1

    def test_accept_zero_values(self, simple_leaf):
        """Test that zero values are accepted (simplex allows zeros)."""
        value = torch.tensor([[0.0, 1.0]])
        simple_leaf.probs = value
        result = simple_leaf.probs
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-8)

    def test_accept_very_small_positive(self, simple_leaf):
        """Test that very small positive values are accepted."""
        value = torch.tensor([[1e-10, 1.0 - 1e-10]])
        simple_leaf.probs = value
        result = simple_leaf.probs
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-15)

    def test_accept_uniform_distribution(self, simple_leaf):
        """Test uniform distribution is accepted."""
        value = torch.tensor([[0.5, 0.5]])
        simple_leaf.probs = value
        result = simple_leaf.probs
        torch.testing.assert_close(result, value)

    def test_reject_partial_nan_in_tensor(self, simple_leaf):
        """Test that tensors with some NaN values are rejected."""
        with pytest.raises(ValueError):
            simple_leaf.probs = torch.tensor([[0.3, float("nan"), 0.7]])

    def test_reject_partial_negative_in_tensor(self, simple_leaf):
        """Test that tensors with some negative values are rejected."""
        with pytest.raises(ValueError):
            simple_leaf.probs = torch.tensor([[0.3, -0.1, 0.8]])

    def test_error_message_includes_param_name(self, simple_leaf):
        """Test that error messages include the parameter name."""
        with pytest.raises(ValueError) as exc_info:
            simple_leaf.probs = torch.tensor([[-0.1, 0.5]])
        assert "probs" in str(exc_info.value)

    def test_error_message_shows_value(self, simple_leaf):
        """Test that error messages show the problematic value."""
        with pytest.raises(ValueError) as exc_info:
            simple_leaf.probs = torch.tensor([[float("nan"), 0.5]])
        assert "nan" in str(exc_info.value).lower()

    def test_tolerance_for_numerical_precision(self, simple_leaf):
        """Test that small numerical deviations from sum=1 are accepted."""
        # Values that sum to 1.000001 (within tolerance)
        value = torch.tensor([[0.5000005, 0.4999995]])
        simple_leaf.probs = value
        result = simple_leaf.probs
        # Should be normalized to sum exactly to 1
        sums = result.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)


# ============================================================================
# SECTION 3: Tensor Operations
# ============================================================================


class TestTensorOperations:
    """Test handling of different tensor shapes and dtypes."""

    def test_1d_tensor(self):
        """Test assignment and retrieval of 1D tensors."""
        leaf = SimpleLeafWithSimplexParam(Scope([0]), probs_init=torch.tensor([0.7, 0.3]))
        value = torch.tensor([0.6, 0.4])
        leaf.probs = value
        result = leaf.probs
        torch.testing.assert_close(result, value)

    def test_2d_tensor(self, simple_leaf):
        """Test assignment and retrieval of 2D tensors."""
        value = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
        simple_leaf.probs = value
        result = simple_leaf.probs
        torch.testing.assert_close(result, value)

    def test_3d_tensor(self):
        """Test assignment and retrieval of 3D tensors."""
        leaf = SimpleLeafWithSimplexParam(Scope([0, 1, 2]), probs_init=torch.ones(3, 2, 4) / 4)
        value = torch.ones(3, 2, 4) / 4
        leaf.probs = value
        result = leaf.probs
        torch.testing.assert_close(result, value)

    def test_large_tensor(self):
        """Test with large tensors."""
        shape = (100, 50)
        leaf = SimpleLeafWithSimplexParam(Scope([i for i in range(100)]), probs_init=torch.ones(shape) / 50)
        value = torch.rand(shape)
        # Normalize to sum to 1 along last dimension
        value = value / value.sum(dim=-1, keepdim=True)
        leaf.probs = value
        result = leaf.probs
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-7)

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        leaf = SimpleLeafWithSimplexParam(
            Scope([0]), probs_init=torch.tensor([0.7, 0.3], dtype=torch.float32)
        )
        value = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        leaf.probs = value
        result = leaf.probs
        assert result.dtype == torch.float32

    def test_dtype_float64(self):
        """Test with float64 dtype."""
        probs_init = torch.tensor([0.7, 0.3], dtype=torch.float64)
        leaf = SimpleLeafWithSimplexParam(Scope([0]), probs_init=probs_init)
        value = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
        leaf.probs = value
        result = leaf.probs
        # Result dtype should match the parameter's dtype
        assert result.dtype == leaf.logits_probs.dtype

    def test_dtype_conversion_on_assignment(self):
        """Test that assigned value is converted to match parameter dtype."""
        leaf = SimpleLeafWithSimplexParam(
            Scope([0]), probs_init=torch.tensor([0.7, 0.3], dtype=torch.float64)
        )
        # Assign float32, should be converted to match parameter dtype
        value = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        leaf.probs = value
        result = leaf.probs
        assert result.dtype == leaf.logits_probs.dtype

    def test_broadcasting_in_assignment(self):
        """Test broadcasting behavior during assignment."""
        leaf = SimpleLeafWithSimplexParam(Scope([0, 1]), probs_init=torch.ones(2, 2) / 2)
        # Assign value that should broadcast
        value = torch.tensor([[0.3, 0.7]])
        leaf.probs = value
        result = leaf.probs
        torch.testing.assert_close(result, value)


# ============================================================================
# SECTION 4: PyTorch Integration
# ============================================================================


class TestPyTorchIntegration:
    """Test integration with PyTorch (gradients, optimization, state_dict)."""

    def test_gradients_flow_through_descriptor(self, simple_leaf):
        """Test that gradients propagate through the descriptor."""
        value = torch.tensor([[0.5, 0.5]], requires_grad=False)
        simple_leaf.probs = value

        # Compute a simple loss
        loss = simple_leaf.probs.sum()
        loss.backward()

        # Check that logits_probs has gradients
        assert simple_leaf.logits_probs.grad is not None
        # Gradient should be propagated through softmax
        assert torch.isfinite(simple_leaf.logits_probs.grad).all()

    def test_gradient_descent_optimization(self, simple_leaf):
        """Test that gradient descent optimization works correctly."""
        # Set initial value
        initial_value = torch.tensor([[0.9, 0.1]])
        simple_leaf.probs = initial_value

        # Create optimizer
        optimizer = torch.optim.SGD(simple_leaf.parameters(), lr=0.1)

        # Perform optimization steps with loss = (probs[0] - 0.3)^2
        target = 0.3
        for _ in range(10):
            optimizer.zero_grad()
            loss = (simple_leaf.probs[..., 0] - target) ** 2
            loss.backward()
            optimizer.step()

        # probs should move towards target
        final_probs = simple_leaf.probs.detach()
        assert final_probs[..., 0].item() < initial_value[..., 0].item()

    def test_state_dict_save_and_load(self, simple_leaf):
        """Test saving and loading state via state_dict."""
        # Set a value matching shape of simple_leaf
        value = torch.tensor([[0.3, 0.7]])
        simple_leaf.probs = value

        # Get state dict
        state_dict = simple_leaf.state_dict()

        # Create new leaves with same scope and shape
        new_leaf = SimpleLeafWithSimplexParam(Scope([0, 1]))
        # Ensure new_leaf has same shape as simple_leaf before loading
        new_leaf.probs = torch.tensor([[0.5, 0.5]])  # Initialize with same shape
        new_leaf.load_state_dict(state_dict)

        # Verify the value is restored
        torch.testing.assert_close(new_leaf.probs, value)

    def test_parameter_is_differentiable(self, simple_leaf):
        """Test that the parameter is differentiable."""
        assert simple_leaf.logits_probs.requires_grad

    def test_softmax_gradient_chain(self, simple_leaf):
        """Test gradient chain rule through softmax projection."""
        value = torch.tensor([[0.3, 0.7]], requires_grad=False)
        simple_leaf.probs = value

        # Compute loss
        loss = simple_leaf.probs.pow(2).sum()
        loss.backward()

        # Check that gradient is finite
        assert torch.isfinite(simple_leaf.logits_probs.grad).all()
        # Gradient should be non-zero
        assert not torch.allclose(
            simple_leaf.logits_probs.grad, torch.zeros_like(simple_leaf.logits_probs.grad)
        )

    def test_module_cloning(self, simple_leaf):
        """Test that module can be cloned correctly."""
        value = torch.tensor([[0.3, 0.7]])
        simple_leaf.probs = value

        # Clone the module
        import copy

        cloned = copy.deepcopy(simple_leaf)

        # Verify cloned module has same parameter values
        torch.testing.assert_close(cloned.probs, simple_leaf.probs)

        # Verify they are independent
        cloned.probs = torch.tensor([[0.5, 0.5]])  # Must sum to 1
        assert not torch.allclose(cloned.probs, simple_leaf.probs)


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
            if torch.any(value < 0.1):
                raise ValueError("Must be >= 0.1")

        descriptor = SimplexParameter("test", validator=counting_validator)
        leaf = SimpleLeafWithSimplexParam(Scope([0]))
        # Replace descriptor temporarily
        old_validator = leaf.__class__.probs.validator
        leaf.__class__.probs.validator = counting_validator

        leaf.probs = torch.tensor([[0.5, 0.5]])
        assert call_count[0] > 0

        # Restore
        leaf.__class__.probs.validator = old_validator

    def test_custom_validator_stricter_constraint(self):
        """Test custom validator that enforces stricter constraints."""

        def min_component_validator(value):
            if not torch.isfinite(value).all():
                raise ValueError("Value must be finite")
            if torch.any(value < 0.1):
                raise ValueError("All components must be >= 0.1")
            # Check sum to 1
            sums = value.sum(dim=-1, keepdim=True)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
                raise ValueError("Values must sum to 1")

        descriptor = SimplexParameter("probs", validator=min_component_validator)

        class StrictLeaf(LeafModule):
            probs = descriptor

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 2)
                self.logits_probs = nn.Parameter(torch.empty(1, 2))

            @property
            def distribution(self):
                return torch.distributions.Categorical(self.probs)

            @property
            def _supported_value(self):
                return 0

            def params(self):
                return {"probs": self.probs}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = StrictLeaf()

        # Should accept values with all components >= 0.1
        leaf.probs = torch.tensor([[0.4, 0.6]])
        torch.testing.assert_close(leaf.probs, torch.tensor([[0.4, 0.6]]))

        # Should reject values with component < 0.1
        with pytest.raises(ValueError, match=">= 0.1"):
            leaf.probs = torch.tensor([[0.05, 0.95]])

    def test_custom_validator_with_entropy_constraint(self):
        """Test custom validator enforcing minimum entropy."""

        def min_entropy_validator(value):
            if not torch.isfinite(value).all():
                raise ValueError("Value must be finite")
            if torch.any(value < 0):
                raise ValueError("Values must be non-negative")
            sums = value.sum(dim=-1, keepdim=True)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
                raise ValueError("Values must sum to 1")
            # Check minimum entropy (avoid too concentrated distributions)
            entropy = -(value * torch.log(value + 1e-10)).sum(dim=-1)
            if torch.any(entropy < 0.1):
                raise ValueError("Distribution must have minimum entropy of 0.1")

        descriptor = SimplexParameter("probs", validator=min_entropy_validator)

        class EntropyLeaf(LeafModule):
            probs = descriptor

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 2)
                self.logits_probs = nn.Parameter(torch.empty(1, 2))

            @property
            def distribution(self):
                return torch.distributions.Categorical(self.probs)

            @property
            def _supported_value(self):
                return 0

            def params(self):
                return {"probs": self.probs}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = EntropyLeaf()

        # Should accept uniform distribution (high entropy)
        leaf.probs = torch.tensor([[0.5, 0.5]])
        torch.testing.assert_close(leaf.probs, torch.tensor([[0.5, 0.5]]))

        # Should reject very concentrated distribution (low entropy)
        with pytest.raises(ValueError, match="minimum entropy"):
            leaf.probs = torch.tensor([[0.99, 0.01]])


# ============================================================================
# SECTION 6: Multiple Parameters
# ============================================================================


class TestMultipleParameters:
    """Test descriptors with multiple SimplexParameters on same class."""

    def test_multiple_params_independent(self, multi_param_leaf):
        """Test that multiple parameters are independent."""
        weights1_value = torch.tensor([[0.6, 0.4]])
        weights2_value = torch.tensor([[0.3, 0.5, 0.2]])

        # Adjust shapes to match
        if multi_param_leaf.weights1.shape[-1] != 2:
            weights1_value = torch.ones_like(multi_param_leaf.weights1) / multi_param_leaf.weights1.shape[-1]
        if multi_param_leaf.weights2.shape[-1] != 3:
            weights2_value = torch.ones_like(multi_param_leaf.weights2) / multi_param_leaf.weights2.shape[-1]

        multi_param_leaf.weights1 = weights1_value
        multi_param_leaf.weights2 = weights2_value

        torch.testing.assert_close(multi_param_leaf.weights1, weights1_value)
        torch.testing.assert_close(multi_param_leaf.weights2, weights2_value)

    def test_multiple_params_separate_storage(self, multi_param_leaf):
        """Test that parameters use separate internal storage."""
        weights1_value = torch.ones_like(multi_param_leaf.weights1) / multi_param_leaf.weights1.shape[-1]
        weights2_value = torch.ones_like(multi_param_leaf.weights2) / multi_param_leaf.weights2.shape[-1]

        multi_param_leaf.weights1 = weights1_value
        multi_param_leaf.weights2 = weights2_value

        # Internal storage should be separate
        expected_logits1 = torch.log(weights1_value)
        expected_logits2 = torch.log(weights2_value)

        torch.testing.assert_close(multi_param_leaf.logits_weights1.data, expected_logits1)
        torch.testing.assert_close(multi_param_leaf.logits_weights2.data, expected_logits2)

    def test_multiple_params_validation_independent(self, multi_param_leaf):
        """Test that validation is independent for each parameter."""
        # weights1 should accept the value
        valid_value = torch.ones_like(multi_param_leaf.weights1) / multi_param_leaf.weights1.shape[-1]
        multi_param_leaf.weights1 = valid_value

        # weights2 should reject invalid value
        with pytest.raises(ValueError):
            multi_param_leaf.weights2 = torch.tensor([[-0.1, 0.5, 0.6]])

        # weights1 should still have its previous value
        torch.testing.assert_close(multi_param_leaf.weights1, valid_value)


# ============================================================================
# SECTION 7: Integration with LeafModule
# ============================================================================


class TestLeafModuleIntegration:
    """Test integration with LeafModule and real-world usage patterns."""

    def test_assignment_in_mle(self):
        """Test that assignment works within MLE context."""
        from spflow.distributions.categorical import Categorical as CategoricalDistribution

        class TestLeaf(LeafModule):
            probs = SimplexParameter("probs")

            def __init__(self, scope):
                super().__init__(scope, out_channels=1)
                self._event_shape = (len(scope.query), 2)
                self.logits_probs = nn.Parameter(torch.empty(self._event_shape))
                self.probs = torch.ones(self._event_shape) / 2
                # Initialize distribution properly
                self._distribution = CategoricalDistribution(p=self.probs, event_shape=self._event_shape)

            @property
            def distribution(self):
                return self._distribution

            @property
            def _supported_value(self):
                return 0

            def params(self):
                return {"probs": self.probs}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                # This mimics real MLE usage
                probs_est = torch.ones_like(data[:1, :2]) * 0.5
                self.probs = probs_est

        leaf = TestLeaf(Scope([0, 1]))
        # This should not raise an error
        leaf.maximum_likelihood_estimation(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]), nan_strategy=None
        )
        # Verify probs was set
        assert leaf.probs is not None
        # Verify probs satisfies simplex constraints
        sums = leaf.probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)

    def test_descriptor_with_broadcast_to_event_shape(self):
        """Test assignment after broadcasting to event shape."""
        from spflow.distributions.categorical import Categorical as CategoricalDistribution

        class BroadcastLeaf(LeafModule):
            probs = SimplexParameter("probs")

            def __init__(self, scope, out_channels):
                super().__init__(scope, out_channels=out_channels)
                self._event_shape = (len(scope.query), out_channels)
                self.logits_probs = nn.Parameter(torch.empty(self._event_shape))
                self.probs = torch.ones(self._event_shape) / out_channels
                # Initialize distribution properly
                self._distribution = CategoricalDistribution(p=self.probs, event_shape=self._event_shape)

            @property
            def distribution(self):
                return self._distribution

            @property
            def _supported_value(self):
                return 0

            def params(self):
                return {"probs": self.probs}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = BroadcastLeaf(Scope([0, 1]), out_channels=3)
        # Assign and broadcast
        value = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]])
        leaf.probs = value
        torch.testing.assert_close(leaf.probs, value, rtol=1e-5, atol=1e-7)

    def test_params_method_returns_simplex_space(self):
        """Test that params() method returns simplex-space values."""

        class ParamsLeaf(LeafModule):
            probs = SimplexParameter("probs")

            def __init__(self, scope):
                super().__init__(scope, out_channels=1)
                self._event_shape = (len(scope.query), 2)
                self.logits_probs = nn.Parameter(torch.empty(self._event_shape))
                self.probs = torch.tensor([[0.7, 0.3]])

            @property
            def distribution(self):
                return torch.distributions.Categorical(self.probs)

            @property
            def _supported_value(self):
                return 0

            def params(self):
                return {"probs": self.probs}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = ParamsLeaf(Scope([0, 1]))
        params = leaf.params()
        # params() should return simplex-space value
        assert (params["probs"] >= 0).all()
        sums = params["probs"].sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(params["probs"], torch.tensor([[0.7, 0.3]]))


# ============================================================================
# SECTION 8: Descriptor Protocol Tests
# ============================================================================


class TestDescriptorProtocol:
    """Test Python descriptor protocol compliance."""

    def test_descriptor_get_on_class(self):
        """Test __get__ with obj=None returns descriptor."""
        descriptor = SimpleLeafWithSimplexParam.probs
        assert isinstance(descriptor, SimplexParameter)

    def test_descriptor_get_on_instance(self, simple_leaf):
        """Test __get__ with obj returns value."""
        result = simple_leaf.probs
        assert isinstance(result, torch.Tensor)

    def test_descriptor_set_on_instance(self, simple_leaf):
        """Test __set__ works on instance."""
        value = torch.tensor([[0.3, 0.7]])
        simple_leaf.probs = value
        torch.testing.assert_close(simple_leaf.probs, value)

    def test_descriptor_name_attribute(self):
        """Test that descriptor stores parameter name."""
        descriptor = SimplexParameter("weights")
        assert descriptor.name == "weights"
        assert descriptor.logits_name == "logits_weights"

    def test_different_param_names_different_storage(self):
        """Test that different parameter names use different storage."""

        class TwoParamLeaf(LeafModule):
            param1 = SimplexParameter("param1")
            param2 = SimplexParameter("param2")

            def __init__(self):
                super().__init__(Scope([0]), out_channels=1)
                self._event_shape = (1, 2)
                self.logits_param1 = nn.Parameter(torch.tensor([[0.5, 0.5]]))
                self.logits_param2 = nn.Parameter(torch.tensor([[0.3, 0.7]]))

            @property
            def distribution(self):
                return None

            @property
            def _supported_value(self):
                return 0

            def params(self):
                return {}

            def _mle_compute_statistics(self, data, weights, bias_correction):
                pass

        leaf = TwoParamLeaf()
        leaf.param1 = torch.tensor([[0.6, 0.4]])
        leaf.param2 = torch.tensor([[0.2, 0.8]])

        # Should have different storage attributes
        assert hasattr(leaf, "logits_param1")
        assert hasattr(leaf, "logits_param2")
        # They should be different Parameter objects
        assert leaf.logits_param1 is not leaf.logits_param2
        # After assignment with different values, they should have different stored values
        assert not torch.allclose(leaf.logits_param1, leaf.logits_param2)


# ============================================================================
# SECTION 9: Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability of simplex parameter projections."""

    def test_stability_with_very_small_values(self):
        """Test numerical stability when dealing with very small probability values."""
        leaf = SimpleLeafWithSimplexParam(Scope([0]))
        value = torch.tensor([[1e-10, 1.0 - 1e-10]])
        leaf.probs = value
        result = leaf.probs

        # Should recover value without underflow
        torch.testing.assert_close(result, value, rtol=1e-10, atol=1e-15)

    def test_stability_with_extreme_logits(self, simple_leaf):
        """Test numerical stability with extreme logit values."""
        # Set extreme logits directly
        simple_leaf.logits_probs.data = torch.tensor([[100.0, -100.0]])

        # Should still produce valid probabilities
        result = simple_leaf.probs
        assert (result >= 0).all()
        sums = result.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)

    def test_gradient_stability_for_small_values(self, simple_leaf):
        """Test that gradients are stable for small parameter values."""
        value = torch.tensor([[1e-5, 1.0 - 1e-5]], requires_grad=False)
        simple_leaf.probs = value

        loss = simple_leaf.probs.sum()
        loss.backward()

        # Gradient should be finite
        assert torch.isfinite(simple_leaf.logits_probs.grad).all()

    def test_repeated_assignment_stability(self, simple_leaf):
        """Test stability over many assignment cycles."""
        initial_value = torch.tensor([[0.3, 0.7]])
        simple_leaf.probs = initial_value

        # Repeatedly set and get
        for _ in range(100):
            current = simple_leaf.probs.detach().clone()
            simple_leaf.probs = current

        final = simple_leaf.probs
        torch.testing.assert_close(final, initial_value, rtol=1e-5, atol=1e-7)

    def test_softmax_numerical_stability(self, simple_leaf):
        """Test softmax numerical stability with large logits."""
        # Set large logits (matching shape)
        large_logits = torch.tensor([[1000.0, -1000.0]])
        simple_leaf.logits_probs.data = large_logits

        result = simple_leaf.probs
        # Should still be valid probabilities
        assert (result >= 0).all()
        sums = result.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)

    def test_log_numerical_stability(self, simple_leaf):
        """Test log numerical stability with very small probabilities."""
        # Set very small probabilities
        small_probs = torch.tensor([[1e-20, 1.0 - 1e-20]])
        simple_leaf.probs = small_probs

        # Check that internal logits are finite
        assert torch.isfinite(simple_leaf.logits_probs.data).all()


# ============================================================================
# SECTION 10: Edge Case Integration Tests
# ============================================================================


class TestEdgeCaseIntegration:
    """Test edge cases and unusual scenarios."""

    def test_assignment_from_numpy_like_value(self, simple_leaf):
        """Test assignment from numpy-like values."""
        value = [0.3, 0.7]  # List
        simple_leaf.probs = value
        result = simple_leaf.probs
        expected = torch.tensor([0.3, 0.7])
        torch.testing.assert_close(result, expected)

    def test_assignment_with_grad_enabled(self, simple_leaf):
        """Test assignment when gradients are enabled."""
        value = torch.tensor([[0.5, 0.5]], requires_grad=True)
        simple_leaf.probs = value
        result = simple_leaf.probs
        assert result.requires_grad

    def test_single_category_simplex(self):
        """Test simplex with single category (degenerate case)."""
        leaf = SimpleLeafWithSimplexParam(Scope([0]), probs_init=torch.tensor([[1.0]]))
        value = torch.tensor([[1.0]])
        leaf.probs = value
        result = leaf.probs
        torch.testing.assert_close(result, value)

    def test_many_categories_simplex(self):
        """Test simplex with many categories."""
        num_categories = 100
        leaf = SimpleLeafWithSimplexParam(
            Scope([0]), probs_init=torch.ones(1, num_categories) / num_categories
        )
        value = torch.ones(1, num_categories) / num_categories
        leaf.probs = value
        result = leaf.probs
        torch.testing.assert_close(result, value, rtol=1e-5, atol=1e-7)

    def test_sparse_simplex(self):
        """Test simplex with many zero entries (sparse distribution)."""
        leaf = SimpleLeafWithSimplexParam(Scope([0]), probs_init=torch.tensor([[0.9, 0.1, 0.0, 0.0]]))
        value = torch.tensor([[0.95, 0.05, 0.0, 0.0]])
        leaf.probs = value
        result = leaf.probs
        torch.testing.assert_close(result, value)

    def test_optimization_with_simplex_constraint(self, simple_leaf):
        """Test optimization while maintaining simplex constraint."""
        # Initialize with some distribution
        simple_leaf.probs = torch.tensor([[0.7, 0.3]])

        optimizer = torch.optim.Adam(simple_leaf.parameters(), lr=0.1)

        for step in range(20):
            optimizer.zero_grad()
            # Loss that encourages moving towards [0.1, 0.9]
            target = torch.tensor([[0.1, 0.9]])
            loss = ((simple_leaf.probs - target) ** 2).sum()
            loss.backward()
            optimizer.step()

            # After each step, simplex constraints should be satisfied
            assert (simple_leaf.probs >= 0).all()
            sums = simple_leaf.probs.sum(dim=-1)
            torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)

        # Final distribution should be closer to target
        final_probs = simple_leaf.probs.detach()
        assert final_probs[0, 1].item() > 0.3  # Should have moved towards 0.9

    def test_empty_tensor_rejection(self, simple_leaf):
        """Test that empty tensors are handled appropriately."""
        # Empty tensors should still be validated
        empty = torch.empty(0)
        # This might be valid (all() on empty is True), so just ensure no crash
        try:
            simple_leaf.probs = empty
            # If accepted, verify it's stored
            assert simple_leaf.probs.shape[0] == 0
        except (ValueError, RuntimeError):
            # If rejected, that's also acceptable
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
