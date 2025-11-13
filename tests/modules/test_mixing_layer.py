from itertools import product

import pytest
import torch

from spflow.modules.rat import MixingLayer
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data

out_channels_values = [1, 5]
out_features_values = [1]
num_repetitions = [7]
params = list(product(out_channels_values, out_features_values, num_repetitions))


def make_sum(in_channels=None, out_channels=None, out_features=None, weights=None, num_repetitions=None):
    if isinstance(weights, list):
        weights = torch.tensor(weights)
        if weights.dim() == 1:
            weights = weights.unsqueeze(1).unsqueeze(2)
        elif weights.dim() == 2:
            weights = weights.unsqueeze(2)

    if weights is not None:
        out_features = weights.shape[0]

    inputs = make_normal_leaf(
        out_features=out_features, out_channels=in_channels, num_repetitions=num_repetitions
    )

    return MixingLayer(
        out_channels=out_channels, inputs=inputs, weights=weights, num_repetitions=num_repetitions
    )


@pytest.mark.parametrize("out_channels,out_features, num_reps", params)
def test_log_likelihood(out_channels: int, out_features: int, num_reps):
    module = make_sum(
        in_channels=out_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    data = make_normal_data(out_features=out_features)
    lls = module.log_likelihood(data)

    assert lls.shape == (data.shape[0], module.out_features, module.out_channels)


@pytest.mark.parametrize("out_channels,out_features,num_reps", params)
def test_sample(out_channels: int, out_features: int, num_reps):
    n_samples = 100
    module = make_sum(
        in_channels=out_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
        mask = torch.full((n_samples, module.out_features), True)
        repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )
        samples = module.sample(data=data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()
