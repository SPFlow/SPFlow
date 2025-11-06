import spflow.meta.data
from tests.fixtures import auto_set_test_seed, auto_set_test_device
import unittest
from itertools import product

from spflow.meta.dispatch import init_default_sampling_context, SamplingContext
from spflow import log_likelihood, sample, marginalize
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_leaf, make_data
from spflow.modules.leaf import Normal, Bernoulli
from spflow.learn import expectation_maximization
from spflow.modules.factorize import Factorize
from spflow.meta.data.scope import Scope
import pytest
import torch

# Constants
in_channels_values = [1, 3]
out_features_values = [4, 8]
num_repetitions = [5]
depth_values = [1, 2]
params = list(product(in_channels_values, out_features_values, num_repetitions, depth_values))


def make_product(in_channels=None, out_features=None, inputs=None, num_repetitions=None, depth=1):
    if inputs is None:
        inputs = make_normal_leaf(
            out_features=out_features, out_channels=in_channels, num_repetitions=num_repetitions
        )
    return Factorize(inputs=[inputs], depth=depth, num_repetitions=num_repetitions)


@pytest.mark.parametrize("in_channels,out_features,num_reps,depth", params)
def test_log_likelihood(in_channels: int, out_features: int, num_reps, depth):
    factorization_layer = make_product(
        in_channels=in_channels, out_features=out_features, num_repetitions=num_reps, depth=depth
    )
    data = make_normal_data(out_features=out_features)
    lls = log_likelihood(factorization_layer, data)
    if num_reps is None:
        assert lls.shape == (
            data.shape[0],
            factorization_layer.out_features,
            factorization_layer.out_channels,
        )
    else:
        assert lls.shape == (
            data.shape[0],
            factorization_layer.out_features,
            factorization_layer.out_channels,
            num_reps,
        )


@pytest.mark.parametrize("in_channels,out_features,num_reps, depth", params)
def test_sample(in_channels: int, out_features: int, num_reps, depth):
    n_samples = 10
    factorization_layer = make_product(
        in_channels=in_channels, out_features=out_features, num_repetitions=num_reps, depth=depth
    )

    data = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(
        low=0, high=factorization_layer.out_channels, size=(n_samples, factorization_layer.out_features)
    )
    mask = torch.full((n_samples, factorization_layer.out_features), True, dtype=torch.bool)
    if num_reps is not None:
        repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    else:
        repetition_index = None
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = sample(factorization_layer, data, sampling_ctx=sampling_ctx)
    assert samples.shape == data.shape
    samples_query = samples[:, factorization_layer.scope.query]
    assert torch.isfinite(samples_query).all()


def test_factorization():
    data = make_normal_data(out_features=4)
    factorization = make_product(in_channels=3, out_features=4, num_repetitions=5)
    factorization = expectation_maximization(factorization, data, max_steps=10)
    assert factorization is not None


@pytest.mark.parametrize(
    "prune,in_channels,marg_rvs,num_reps",
    product(
        [True, False],
        in_channels_values,
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        num_repetitions,
    ),
)
def test_marginalize(prune, in_channels: int, marg_rvs: list[int], num_reps):
    out_features = 6
    module = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)

    # Marginalize scope
    marginalized_module = marginalize(module, marg_rvs, prune=prune)

    if len(marg_rvs) == out_features:
        assert marginalized_module is None
        return

    if prune and len(marg_rvs) == (out_features - 1):
        # If pruning is active and only one scope is left, the (pruned) input module should be returned
        assert isinstance(marginalized_module, type(module.inputs))

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


def test_multidistribution_input():
    out_channels = 3
    num_reps = 5
    out_features_1 = 8
    out_features_2 = 10

    scope_1 = Scope(list(range(0, out_features_1)))
    scope_2 = Scope(list(range(out_features_1, out_features_1 + out_features_2)))

    cls_1 = Normal
    cls_2 = Bernoulli

    module_1 = make_leaf(cls=cls_1, out_channels=out_channels, scope=scope_1, num_repetitions=num_reps)
    data_1 = make_data(cls=cls_1, out_features=out_features_1, n_samples=5)

    module_2 = make_leaf(cls=cls_2, out_channels=out_channels, scope=scope_2, num_repetitions=num_reps)
    data_2 = make_data(cls=cls_2, out_features=out_features_2, n_samples=5)

    module = Factorize(inputs=[module_1, module_2], depth=2, num_repetitions=num_reps)

    data = torch.cat((data_1, data_2), dim=1)
    lls = log_likelihood(module, data)

    assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)

    repetition_idx = torch.zeros((1,), dtype=torch.long)
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, num_samples=1)
    sampling_ctx.repetition_idx = repetition_idx
    samples = sample(module, sampling_ctx=sampling_ctx)

    assert samples.shape == (1, out_features_1 + out_features_2)
