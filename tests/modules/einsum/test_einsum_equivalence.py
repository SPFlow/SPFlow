"""Tests for EinsumLayer equivalence with SumLayer(OuterProductLayer)."""

from itertools import product
import pytest
import torch
import numpy as np

from spflow.meta import Scope
from spflow.modules.einsum import EinsumLayer
from spflow.modules.sums import Sum
from spflow.modules.products import OuterProduct
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data, DummyLeaf, make_leaf


# Test parameter combinations
in_channels_values = [1, 3]
out_channels_values = [1, 4]
in_features_values = [2, 4]  # D (Must be even for splitting if using simple split logic)
num_repetitions_values = [1, 2]

params = list(
    product(in_channels_values, out_channels_values, in_features_values, num_repetitions_values)
)


class TestEinsumLayerEquivalence:
    """Test EinsumLayer equivalence with SumLayer(OuterProductLayer)."""
    
    def _create_models(self, in_channels, out_channels, in_features, num_reps):
        """Helper to create equivalent models and sync weights."""
        half_features = in_features // 2
        
        left_input = make_normal_leaf(
            scope=list(range(0, half_features)),
            out_features=half_features, 
            out_channels=in_channels, 
            num_repetitions=num_reps
        )
        right_input = make_normal_leaf(
            scope=list(range(half_features, in_features)),
            out_features=half_features, 
            out_channels=in_channels, 
            num_repetitions=num_reps
        )
        
        einsum = EinsumLayer(
            inputs=[left_input, right_input],
            out_channels=out_channels,
            num_repetitions=num_reps
        )
        
        prod_layer = OuterProduct(
            inputs=[left_input, right_input],
            num_splits=2
        )
        
        sum_layer = Sum(
            inputs=prod_layer,
            out_channels=out_channels,
            num_repetitions=num_reps
        )
        
        # Sync Weights
        # (D, O, R, I, J) -> (D, I*J, O, R)
        w_einsum = einsum.weights
        w_permuted = w_einsum.permute(0, 3, 4, 1, 2)
        w_sum = w_permuted.reshape(half_features, in_channels * in_channels, out_channels, num_reps)
        sum_layer.weights = w_sum
        
        return einsum, sum_layer

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_log_likelihood_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test log-likelihood equivalence."""
        einsum, sum_layer = self._create_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        data = make_normal_data(out_features=in_features, num_samples=batch_size)
        
        ll_einsum = einsum.log_likelihood(data)
        ll_sum = sum_layer.log_likelihood(data)
        
        assert torch.allclose(ll_einsum, ll_sum, atol=1e-6), \
            f"Log-likelihoods mismatch. Max diff: {(ll_einsum - ll_sum).abs().max()}"

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_sampling_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test sampling equivalence with fixed seed."""
        einsum, sum_layer = self._create_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        
        torch.manual_seed(42)
        # Note: einsum vs outer_product feature mismatch in out_shape?
        # EinsumLayer([left, right]) has features = half_features (if split) or features (if inputs is list)?
        # Let's check EinsumLayer init logic.
        # if input is list: out_shape.features = inputs[0].out_shape.features (= half_features)
        # So we use half_features for shape access.
        
        out_features = einsum.out_shape.features
        
        channel_indices = torch.randint(0, out_channels, (batch_size, out_features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, out_features), dtype=torch.bool)
        
        ctx_common = SamplingContext(
            channel_index=channel_indices,
            repetition_index=repetition_indices,
            mask=mask
        )
        
        # Sample from Einsum
        torch.manual_seed(42)
        sample_einsum = einsum.sample(num_samples=batch_size, sampling_ctx=ctx_common.copy())
        
        # Sample from Sum(Prod)
        torch.manual_seed(42)
        sample_sum = sum_layer.sample(num_samples=batch_size, sampling_ctx=ctx_common.copy())
        
        assert torch.allclose(sample_einsum, sample_sum), \
            "Samples mismatch despite fixed seed."
