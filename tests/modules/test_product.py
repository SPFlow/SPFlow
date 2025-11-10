from tests.fixtures import auto_set_test_seed, auto_set_test_device
import unittest
from itertools import product

from spflow.meta import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data
from spflow.learn import expectation_maximization
from spflow.modules import Product
import pytest
import torch

# Constants
in_channels_values = [1, 3]
out_features_values = [1, 4]
num_repetitions = [None, 5]
params = list(product(in_channels_values, out_features_values, num_repetitions))


def make_product(in_channels=None, out_features=None, inputs=None, num_repetitions=None):
    if inputs is None:
        inputs = make_normal_leaf(
            out_features=out_features, out_channels=in_channels, num_repetitions=num_repetitions
        )
    return Product(inputs=inputs)


@pytest.mark.parametrize("in_channels,out_features,num_reps", params)
def test_log_likelihood(in_channels: int, out_features: int, num_reps, device):
    product_layer = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    data = make_normal_data(out_features=out_features)
    lls = product_layer.log_likelihood(data)
    if num_reps is None:
        assert lls.shape == (data.shape[0], 1, product_layer.out_channels)
    else:
        assert lls.shape == (data.shape[0], 1, product_layer.out_channels, num_reps)


@pytest.mark.parametrize("in_channels,out_features,num_reps", params)
def test_sample(in_channels: int, out_features: int, num_reps, device):
    n_samples = 10
    product_layer = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    for i in range(product_layer.out_channels):
        data = torch.full((n_samples, out_features), torch.nan)
        channel_index = torch.full((n_samples, out_features), fill_value=i)
        mask = torch.full((n_samples, out_features), True, dtype=torch.bool)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )
        samples = product_layer.sample(data=data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, product_layer.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("in_channels,out_features,num_reps", params)
def test_expectation_maximization(in_channels: int, out_features: int, num_reps, device):
    product_layer = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    data = make_normal_data(out_features=out_features)
    with torch.autograd.set_detect_anomaly(True):
        expectation_maximization(product_layer, data, max_steps=10)


def test_constructor():
    pass


@pytest.mark.parametrize(
    "prune,in_channels,marg_rvs,num_reps",
    product(
        [True, False],
        in_channels_values,
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        num_repetitions,
    ),
)
def test_marginalize(prune, in_channels: int, marg_rvs: list[int], num_reps, device):
    out_features = 3
    module = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)

    # Marginalize scope
    marginalized_module = module.marginalize(marg_rvs, prune=prune)

    if len(marg_rvs) == out_features:
        assert marginalized_module is None
        return

    if prune and len(marg_rvs) == (out_features - 1):
        # If pruning is active and only one scope is left, the (pruned) input module should be returned
        assert isinstance(marginalized_module, type(module.inputs))

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


def test_multiple_inputs():
    in_channels = 2
    out_channels = 2
    out_features = 4
    num_reps = 5

    mean = torch.rand((out_features, out_channels, num_reps))
    std = torch.rand((out_features, out_channels, num_reps))

    normal_layer_a = make_normal_leaf(
        scope=[0, 1, 2, 3],
        out_features=out_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
        mean=mean,
        std=std,
    )
    normal_layer_b1 = make_normal_leaf(
        scope=[0, 1],
        out_features=out_features / 2,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
        mean=mean[0:2, :, :],
        std=std[0:2, :, :],
    )
    normal_layer_b2 = make_normal_leaf(
        scope=[2, 3],
        out_features=out_features / 2,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
        mean=mean[2:4, :, :],
        std=std[2:4, :, :],
    )

    module_a = Product(inputs=normal_layer_a)

    module_b = Product(inputs=[normal_layer_b1, normal_layer_b2])

    # test log likelihood

    data = make_normal_data(out_features=out_features)

    ll_a = module_a.log_likelihood(data)
    ll_b = module_b.log_likelihood(data)

    assert torch.allclose(ll_a, ll_b)

    # test sampling

    n_samples = 10

    data_a = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(low=0, high=out_channels, size=(n_samples, out_features))
    mask = torch.full((n_samples, out_features), True)
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index, mask=mask, repetition_index=repetition_index
    )

    data_b = torch.full((n_samples, out_features), torch.nan)

    sampling_ctx_b = SamplingContext(
        channel_index=channel_index, mask=mask, repetition_index=repetition_index
    )

    samples_a = module_a.sample(data=data_a, is_mpe=True, sampling_ctx=sampling_ctx_a)
    samples_b = module_b.sample(data=data_b, is_mpe=True, sampling_ctx=sampling_ctx_b)

    assert torch.allclose(samples_a, samples_b)
