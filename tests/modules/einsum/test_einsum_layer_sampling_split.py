"""Sampling and split-related tests for EinsumLayer."""

from itertools import product

import pytest
import torch

from spflow.modules.einsum import EinsumLayer
from spflow.modules.ops.split import Split
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.utils.sampling_context import SamplingContext
from tests.modules.einsum.layer_test_utils import make_einsum_single_input, make_einsum_two_inputs
from tests.utils.leaves import make_normal_data, make_normal_leaf


class TestEinsumLayerConstruction:
    """Construction validation tests not covered by contracts."""

    def test_invalid_odd_features(self):
        inputs = make_normal_leaf(out_features=3, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            EinsumLayer(inputs=inputs, out_channels=2)


class TestEinsumLayerSampling:
    """Test EinsumLayer sampling."""

    @pytest.mark.parametrize(
        "in_channels,out_channels,in_features,num_reps",
        product([2], [3], [4], [1, 2]),
    )
    def test_sample_two_inputs(self, in_channels: int, out_channels: int, in_features: int, num_reps: int):
        num_samples = 50
        module = make_einsum_two_inputs(in_channels, out_channels, in_features, num_reps)
        total_features = in_features * 2

        data = torch.full((num_samples, total_features), torch.nan)
        channel_index = torch.randint(low=0, high=out_channels, size=(num_samples, module.out_shape.features))
        mask = torch.ones((num_samples, module.out_shape.features), dtype=torch.bool)
        repetition_index = torch.randint(low=0, high=num_reps, size=(num_samples,))

        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )

        samples = module.sample(data=data, sampling_ctx=sampling_ctx)

        assert samples.shape == (num_samples, total_features)
        assert torch.isfinite(samples[:, module.scope.query]).all()

    def test_mpe_sampling(self):
        num_samples = 20
        module = make_einsum_single_input(2, 3, 4, 1)

        data = torch.full((num_samples, 4), torch.nan)
        channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
        mask = torch.ones((num_samples, 2), dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

        samples = module.sample(data=data, is_mpe=True, sampling_ctx=sampling_ctx)

        assert samples.shape == (num_samples, 4)
        assert torch.isfinite(samples).all()


class TestEinsumLayerSplitOptimization:
    """Test that EinsumLayer reuses Split modules when passed directly."""

    def test_split_input_not_wrapped(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        split_mode = SplitConsecutive(leaf)

        einsum = EinsumLayer(inputs=split_mode, out_channels=3)

        assert einsum.inputs is split_mode
        assert isinstance(einsum.inputs, Split)
        assert not isinstance(einsum.inputs.inputs, Split)

    def test_split_input_produces_same_output(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        data = make_normal_data(out_features=4)

        split_mode = SplitConsecutive(leaf)
        einsum_wrapped = EinsumLayer(inputs=leaf, out_channels=3)
        einsum_direct = EinsumLayer(inputs=split_mode, out_channels=3)

        einsum_direct.logits.data = einsum_wrapped.logits.data.clone()

        lls_wrapped = einsum_wrapped.log_likelihood(data)
        lls_direct = einsum_direct.log_likelihood(data)

        torch.testing.assert_close(lls_wrapped, lls_direct, rtol=1e-5, atol=1e-8)

    def test_split_sampling_works(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        split_mode = SplitConsecutive(leaf)
        einsum = EinsumLayer(inputs=split_mode, out_channels=3)

        num_samples = 20
        data = torch.full((num_samples, 4), torch.nan)
        channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
        mask = torch.ones((num_samples, 2), dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

        samples = einsum.sample(data=data, sampling_ctx=sampling_ctx)

        assert samples.shape == (num_samples, 4)
        assert torch.isfinite(samples).all()

    def test_split_alternate_input_works(self):
        from spflow.modules.ops.split_interleaved import SplitInterleaved

        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        split_alt = SplitInterleaved(leaf)
        einsum = EinsumLayer(inputs=split_alt, out_channels=3)

        assert einsum.inputs is split_alt
        assert isinstance(einsum.inputs, Split)

        data = make_normal_data(out_features=4)
        lls = einsum.log_likelihood(data)
        assert torch.isfinite(lls).all()

        num_samples = 20
        sample_data = torch.full((num_samples, 4), torch.nan)
        channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
        mask = torch.ones((num_samples, 2), dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

        samples = einsum.sample(data=sample_data, sampling_ctx=sampling_ctx)
        assert samples.shape == (num_samples, 4)
        assert torch.isfinite(samples).all()
