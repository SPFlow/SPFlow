"""Tests for LinsumLayer equivalence with SumLayer(ElementwiseProductLayer)."""

from itertools import product
import pytest
import torch
import numpy as np

from spflow.meta import Scope
from spflow.modules.einsum import LinsumLayer
from spflow.modules.sums import Sum
from spflow.modules.products import ElementwiseProduct
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data, DummyLeaf, make_leaf


# Test parameter combinations
in_channels_values = [1, 3]
out_channels_values = [1, 4]
in_features_values = [2, 4]  # Must be even for splitting
num_repetitions_values = [1, 2]

params = list(
    product(in_channels_values, out_channels_values, in_features_values, num_repetitions_values)
)


class TestLinsumLayerEquivalence:
    """Test LinsumLayer equivalence with SumLayer(ElementwiseProductLayer)."""
    
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
        
        # Create LinsumLayer
        linsum = LinsumLayer(
            inputs=[left_input, right_input],
            out_channels=out_channels,
            num_repetitions=num_reps
        )
        
        # Create Sum(ElementwiseProduct) Equivalent
        prod_layer = ElementwiseProduct(
            inputs=[left_input, right_input],
            num_splits=2
        )
        
        sum_layer = Sum(
            inputs=prod_layer,
            out_channels=out_channels,
            num_repetitions=num_reps
        )
        
        # Sync Weights
        # Permutation: (D, O, R, C) -> (D, C, O, R)
        # 0 -> 0 (D), 3 -> 1 (C), 1 -> 2 (O), 2 -> 3 (R)
        w_linsum = linsum.weights
        w_sum = w_linsum.permute(0, 3, 1, 2)
        sum_layer.weights = w_sum
        
        return linsum, sum_layer

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_log_likelihood_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test log-likelihood equivalence."""
        linsum, sum_layer = self._create_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        data = make_normal_data(out_features=in_features, num_samples=batch_size)
        
        ll_linsum = linsum.log_likelihood(data)
        ll_sum = sum_layer.log_likelihood(data)
        
        assert torch.allclose(ll_linsum, ll_sum, atol=1e-6), \
            f"Log-likelihoods mismatch. Max diff: {(ll_linsum - ll_sum).abs().max()}"

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_sampling_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test sampling equivalence with fixed seed."""
        linsum, sum_layer = self._create_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        
        # Common Sampling Context
        torch.manual_seed(42)
        channel_indices = torch.randint(0, out_channels, (batch_size, linsum.out_shape.features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, linsum.out_shape.features), dtype=torch.bool)
        
        ctx_common = SamplingContext(
            channel_index=channel_indices,
            repetition_index=repetition_indices,
            mask=mask
        )
        
        # Sample from Linsum
        torch.manual_seed(42)
        sample_linsum = linsum.sample(num_samples=batch_size, sampling_ctx=ctx_common.copy())
        
        # Sample from Sum(Prod)
        torch.manual_seed(42)
        sample_sum = sum_layer.sample(num_samples=batch_size, sampling_ctx=ctx_common.copy())
        
        assert torch.allclose(sample_linsum, sample_sum), \
            "Samples mismatch despite fixed seed."


class TestLinsumLayerSingleInputEquivalence:
    """Test LinsumLayer single-input (SplitConsecutive) equivalence with Sum(ElementwiseProduct(SplitConsecutive))."""
    
    # Filter out configurations where in_channels > out_features // 2 
    # (causes ElementwiseProduct shape validation issues unrelated to LinsumLayer)
    single_input_params = [
        (ic, oc, if_, nr) for ic, oc, if_, nr in params 
        if ic <= if_ // 2
    ]
    
    def _create_single_input_models(self, in_channels, out_channels, in_features, num_reps):
        """Helper to create equivalent models with single input and sync weights."""
        # Create single input that will be split internally
        from spflow.modules.ops.split_consecutive import SplitConsecutive
        
        leaf = make_normal_leaf(
            scope=list(range(in_features)),
            out_features=in_features, 
            out_channels=in_channels, 
            num_repetitions=num_reps
        )
        
        # Create LinsumLayer with single input (uses internal SplitConsecutive)
        linsum = LinsumLayer(
            inputs=leaf,
            out_channels=out_channels,
            num_repetitions=num_reps
        )
        
        # Create Sum(ElementwiseProduct(SplitConsecutive)) Equivalent
        split_layer = SplitConsecutive(leaf, dim=1, num_splits=2)
        prod_layer = ElementwiseProduct(inputs=split_layer, num_splits=2)
        sum_layer = Sum(
            inputs=prod_layer,
            out_channels=out_channels,
            num_repetitions=num_reps
        )
        
        # Sync Weights
        # LinsumLayer: (D, O, R, C) -> Sum: (D, C, O, R)
        w_linsum = linsum.weights
        w_sum = w_linsum.permute(0, 3, 1, 2)
        sum_layer.weights = w_sum
        
        return linsum, sum_layer

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", single_input_params)
    def test_log_likelihood_single_input_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test log-likelihood equivalence for single-input LinsumLayer."""
        linsum, sum_layer = self._create_single_input_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        data = make_normal_data(out_features=in_features, num_samples=batch_size)
        
        ll_linsum = linsum.log_likelihood(data)
        ll_sum = sum_layer.log_likelihood(data)
        
        assert torch.allclose(ll_linsum, ll_sum, atol=1e-6), \
            f"Log-likelihoods mismatch. Max diff: {(ll_linsum - ll_sum).abs().max()}"

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", single_input_params)
    def test_sampling_single_input_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test sampling equivalence for single-input LinsumLayer with fixed seed."""
        linsum, sum_layer = self._create_single_input_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        
        # Common Sampling Context
        torch.manual_seed(42)
        channel_indices = torch.randint(0, out_channels, (batch_size, linsum.out_shape.features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, linsum.out_shape.features), dtype=torch.bool)
        
        ctx_common = SamplingContext(
            channel_index=channel_indices,
            repetition_index=repetition_indices,
            mask=mask
        )
        
        # Sample from Linsum
        torch.manual_seed(42)
        sample_linsum = linsum.sample(num_samples=batch_size, sampling_ctx=ctx_common.copy())
        
        # Sample from Sum(Prod)
        torch.manual_seed(42)
        sample_sum = sum_layer.sample(num_samples=batch_size, sampling_ctx=ctx_common.copy())
        
        assert torch.allclose(sample_linsum, sample_sum), \
            "Samples mismatch despite fixed seed."

