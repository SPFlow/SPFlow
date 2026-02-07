"""Sampling and split-related tests for LinsumLayer."""

from itertools import product

import pytest
import torch

from spflow.modules.einsum import LinsumLayer
from spflow.modules.ops.split import Split
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.modules.ops.split_interleaved import SplitInterleaved
from spflow.utils.sampling_context import SamplingContext
from tests.modules.einsum.layer_test_utils import make_linsum_single_input, make_linsum_two_inputs
from tests.utils.leaves import make_normal_data, make_normal_leaf


class TestLinsumLayerConstruction:
    def test_invalid_odd_features(self):
        inputs = make_normal_leaf(out_features=3, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            LinsumLayer(inputs=inputs, out_channels=2)


class TestLinsumLayerSampling:
    @pytest.mark.parametrize(
        "in_channels,out_channels,in_features,num_reps",
        product([2], [3], [4], [1, 2]),
    )
    def test_sample_two_inputs(self, in_channels: int, out_channels: int, in_features: int, num_reps: int):
        num_samples = 50
        module = make_linsum_two_inputs(in_channels, out_channels, in_features, num_reps)
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
        module = make_linsum_single_input(2, 3, 4, 1)

        data = torch.full((num_samples, 4), torch.nan)
        channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
        mask = torch.ones((num_samples, 2), dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

        samples = module.sample(data=data, is_mpe=True, sampling_ctx=sampling_ctx)

        assert samples.shape == (num_samples, 4)
        assert torch.isfinite(samples).all()


class TestLinsumLayerSplitConsecutiveOptimization:
    def test_split_mode_input_not_wrapped(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        split_mode = SplitConsecutive(leaf)

        linsum = LinsumLayer(inputs=split_mode, out_channels=3)

        assert linsum.inputs is split_mode
        assert isinstance(linsum.inputs, Split)
        assert not isinstance(linsum.inputs.inputs, Split)

    def test_split_mode_input_produces_same_output(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        data = make_normal_data(out_features=4)

        split_mode = SplitConsecutive(leaf)
        linsum_wrapped = LinsumLayer(inputs=leaf, out_channels=3)
        linsum_direct = LinsumLayer(inputs=split_mode, out_channels=3)

        linsum_direct.logits.data = linsum_wrapped.logits.data.clone()

        lls_wrapped = linsum_wrapped.log_likelihood(data)
        lls_direct = linsum_direct.log_likelihood(data)

        torch.testing.assert_close(lls_wrapped, lls_direct, rtol=1e-5, atol=1e-8)

    def test_split_mode_sampling_works(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        split_mode = SplitConsecutive(leaf)
        linsum = LinsumLayer(inputs=split_mode, out_channels=3)

        num_samples = 20
        data = torch.full((num_samples, 4), torch.nan)
        channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
        mask = torch.ones((num_samples, 2), dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

        samples = linsum.sample(data=data, sampling_ctx=sampling_ctx)

        assert samples.shape == (num_samples, 4)
        assert torch.isfinite(samples).all()

    def test_split_alternate_input_works(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        split_alt = SplitInterleaved(leaf)
        linsum = LinsumLayer(inputs=split_alt, out_channels=3)

        assert linsum.inputs is split_alt
        assert isinstance(linsum.inputs, Split)

        data = make_normal_data(out_features=4)
        lls = linsum.log_likelihood(data)
        assert torch.isfinite(lls).all()

        num_samples = 20
        sample_data = torch.full((num_samples, 4), torch.nan)
        channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
        mask = torch.ones((num_samples, 2), dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

        samples = linsum.sample(data=sample_data, sampling_ctx=sampling_ctx)
        assert samples.shape == (num_samples, 4)
        assert torch.isfinite(samples).all()
