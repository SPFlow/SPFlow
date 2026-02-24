from itertools import product

import pytest
import torch

from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import make_normal_leaf, make_normal_data
from tests.utils.sampling_context_helpers import patch_simple_as_categorical_one_hot

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

    return RepetitionMixingLayer(
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

    num_reps_after_mixing = 1
    assert lls.shape == (
        data.shape[0],
        module.out_shape.features,
        module.out_shape.channels,
        num_reps_after_mixing,
    )


@pytest.mark.parametrize("out_channels,out_features,num_reps", params)
def test_sample(out_channels: int, out_features: int, num_reps):
    n_samples = 100
    module = make_sum(
        in_channels=out_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    data = torch.full((n_samples, module.out_shape.features), torch.nan)
    channel_index = torch.randint(
        low=0, high=module.out_shape.channels, size=(n_samples, module.out_shape.features)
    )
    mask = torch.full((n_samples, module.out_shape.features), True)
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("out_channels,out_features,num_reps", params)
def test_sample_differentiable(out_channels: int, out_features: int, num_reps):
    n_samples = 64
    module = make_sum(
        in_channels=out_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    channel_index = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(n_samples, module.out_shape.features),
    )
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=module.out_shape.channels),
        mask=torch.ones((n_samples, module.out_shape.features), dtype=torch.bool),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_differentiable=True,
        hard=True,
    )
    out = module._sample(
        data=torch.full((n_samples, module.out_shape.features), torch.nan),
        sampling_ctx=sampling_ctx,
        cache=Cache(),
    )
    assert out.shape == (n_samples, module.out_shape.features)
    assert torch.isfinite(out[:, module.scope.query]).all()
    assert sampling_ctx.repetition_index.shape == (n_samples, num_reps)
    torch.testing.assert_close(
        sampling_ctx.repetition_index.sum(dim=-1),
        torch.ones((n_samples,), dtype=sampling_ctx.repetition_index.dtype),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("out_channels,out_features,num_reps", params)
def test_sample_differentiable_with_conditional_cache(out_channels: int, out_features: int, num_reps):
    n_samples = 48
    module = make_sum(
        in_channels=out_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    cache = Cache()
    module.log_likelihood(torch.randn(n_samples, out_features), cache=cache)
    channel_index = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(n_samples, module.out_shape.features),
    )
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=module.out_shape.channels),
        mask=torch.ones((n_samples, module.out_shape.features), dtype=torch.bool),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_differentiable=True,
        hard=True,
    )
    out = module._sample(
        data=torch.full((n_samples, module.out_shape.features), torch.nan),
        sampling_ctx=sampling_ctx,
        cache=cache,
    )
    assert out.shape == (n_samples, module.out_shape.features)
    assert torch.isfinite(out[:, module.scope.query]).all()


def test_sample_differentiable_equals_non_diff_sampling(monkeypatch: pytest.MonkeyPatch):
    n_samples = 32
    out_channels = 3
    out_features = 1
    num_reps = 5
    module = make_sum(
        in_channels=out_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    channel_index = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(n_samples, module.out_shape.features),
    )
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    mask = torch.ones((n_samples, module.out_shape.features), dtype=torch.bool)
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index.clone(),
        mask=mask.clone(),
        repetition_index=repetition_index.clone(),
        is_mpe=False,
    )
    sampling_ctx_b = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=module.out_shape.channels),
        mask=mask.clone(),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_mpe=False,
        is_differentiable=True,
        hard=True,
    )

    patch_simple_as_categorical_one_hot(monkeypatch)

    torch.manual_seed(1337)
    samples_a = module._sample(
        data=torch.full((n_samples, module.out_shape.features), torch.nan),
        sampling_ctx=sampling_ctx_a,
        cache=Cache(),
    )
    torch.manual_seed(1337)
    samples_b = module._sample(
        data=torch.full((n_samples, module.out_shape.features), torch.nan),
        sampling_ctx=sampling_ctx_b,
        cache=Cache(),
    )

    torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        sampling_ctx_b.repetition_index,
        to_one_hot(sampling_ctx_a.repetition_index, dim=-1, dim_size=num_reps),
        rtol=0.0,
        atol=0.0,
    )


def test_sample_differentiable_equals_non_diff_sampling_with_conditional_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    n_samples = 24
    out_channels = 3
    out_features = 1
    num_reps = 4
    module = make_sum(
        in_channels=out_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )

    evidence = torch.randn(n_samples, out_features)
    cache_a = Cache()
    cache_b = Cache()
    module.log_likelihood(evidence, cache=cache_a)
    module.log_likelihood(evidence, cache=cache_b)

    channel_index = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(n_samples, module.out_shape.features),
    )
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    mask = torch.ones((n_samples, module.out_shape.features), dtype=torch.bool)
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index.clone(),
        mask=mask.clone(),
        repetition_index=repetition_index.clone(),
    )
    sampling_ctx_b = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=module.out_shape.channels),
        mask=mask.clone(),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_differentiable=True,
        hard=True,
    )

    patch_simple_as_categorical_one_hot(monkeypatch)

    torch.manual_seed(1337)
    samples_a = module._sample(
        data=torch.full((n_samples, module.out_shape.features), torch.nan),
        sampling_ctx=sampling_ctx_a,
        cache=cache_a,
    )
    torch.manual_seed(1337)
    samples_b = module._sample(
        data=torch.full((n_samples, module.out_shape.features), torch.nan),
        sampling_ctx=sampling_ctx_b,
        cache=cache_b,
    )

    torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        sampling_ctx_b.repetition_index,
        to_one_hot(sampling_ctx_a.repetition_index, dim=-1, dim_size=num_reps),
        rtol=0.0,
        atol=0.0,
    )
