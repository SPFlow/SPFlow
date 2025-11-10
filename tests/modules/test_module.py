"""Test cases for the base Module class."""

import pytest
import torch
from torch import Tensor

from spflow.meta.data.scope import Scope
from spflow.modules.leaf import Normal
from spflow.utils.cache import Cache


class ConcreteModule(Normal):
    """Concrete implementation of Module for testing _prepare_sample_data."""

    def __init__(self, scope: Scope, out_channels: int = 1):
        super().__init__(scope=scope, out_channels=out_channels)


class TestPrepareSampleData:
    """Test cases for Module._prepare_sample_data method."""

    def test_both_provided_matching_shapes(self):
        """Test when both num_samples and data are provided with matching shapes."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        num_samples = 10
        data = torch.randn(num_samples, 3)  # 3 features matching scope length

        result = module._prepare_sample_data(num_samples, data)

        assert result is data
        assert result.shape == (10, 3)

    def test_both_provided_non_matching_shapes(self):
        """Test when both num_samples and data are provided with non-matching shapes."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        num_samples = 10
        data = torch.randn(5, 3)  # 5 samples != num_samples

        with pytest.raises(ValueError) as exc_info:
            module._prepare_sample_data(num_samples, data)

        assert "num_samples (10) must match data.shape[0] (5)" in str(exc_info.value)

    def test_only_data_provided(self):
        """Test when only data is provided (num_samples=None)."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        data = torch.randn(15, 3)

        result = module._prepare_sample_data(None, data)

        assert result is data
        assert result.shape == (15, 3)

    def test_only_num_samples_provided(self):
        """Test when only num_samples is provided (data=None)."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        num_samples = 20

        result = module._prepare_sample_data(num_samples, None)

        assert result.shape == (20, 3)  # 3 features from scope
        assert torch.all(torch.isnan(result))

    def test_neither_provided(self):
        """Test when neither num_samples nor data is provided."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)

        result = module._prepare_sample_data(None, None)

        assert result.shape == (1, 3)  # Defaults to 1 sample
        assert torch.all(torch.isnan(result))

    def test_num_samples_zero_with_data(self):
        """Test when num_samples=0 and data is provided."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        data = torch.randn(0, 3)  # Empty data

        result = module._prepare_sample_data(0, data)

        assert result is data
        assert result.shape == (0, 3)

    def test_num_samples_zero_without_data(self):
        """Test when num_samples=0 and data is None."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)

        result = module._prepare_sample_data(0, None)

        assert result.shape == (0, 3)
        assert torch.all(torch.isnan(result))

    def test_device_cpu(self):
        """Test that created tensor is on CPU device when module is on CPU."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        module.to("cpu")

        result = module._prepare_sample_data(5, None)

        assert result.device.type == "cpu"
        assert result.shape == (5, 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        """Test that created tensor is on CUDA device when module is on CUDA."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        module.to("cuda")

        result = module._prepare_sample_data(5, None)

        assert result.device.type == "cuda"
        assert result.shape == (5, 3)

    def test_single_feature_scope(self):
        """Test with a single feature in scope."""
        module = ConcreteModule(scope=Scope([0]), out_channels=1)

        result = module._prepare_sample_data(3, None)

        assert result.shape == (3, 1)
        assert torch.all(torch.isnan(result))

    def test_large_scope(self):
        """Test with a large scope."""
        module = ConcreteModule(scope=Scope(list(range(100))), out_channels=2)

        result = module._prepare_sample_data(10, None)

        assert result.shape == (10, 100)
        assert torch.all(torch.isnan(result))

    def test_data_with_existing_values(self):
        """Test that existing data values are preserved."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = module._prepare_sample_data(2, data)

        assert result is data
        assert torch.allclose(result, data)

    def test_data_with_mixed_nan_values(self):
        """Test that mixed NaN values in data are preserved."""
        module = ConcreteModule(scope=Scope([0, 1, 2]), out_channels=2)
        data = torch.tensor([[1.0, float("nan"), 3.0], [float("nan"), 5.0, 6.0]])

        result = module._prepare_sample_data(2, data)

        assert result is data
        assert torch.isnan(result[0, 1])
        assert torch.isnan(result[1, 0])
        assert result[0, 0] == 1.0
        assert result[0, 2] == 3.0

    def test_large_num_samples(self):
        """Test with a large number of samples."""
        module = ConcreteModule(scope=Scope([0, 1]), out_channels=1)
        num_samples = 10000

        result = module._prepare_sample_data(num_samples, None)

        assert result.shape == (10000, 2)
        assert torch.all(torch.isnan(result))

    def test_error_message_format(self):
        """Test that error message includes correct values."""
        module = ConcreteModule(scope=Scope([0, 1]), out_channels=1)
        num_samples = 100
        data = torch.randn(50, 2)

        with pytest.raises(ValueError) as exc_info:
            module._prepare_sample_data(num_samples, data)

        error_msg = str(exc_info.value)
        assert "100" in error_msg
        assert "50" in error_msg
        assert "num_samples" in error_msg
        assert "data.shape[0]" in error_msg
