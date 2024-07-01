from tests.fixtures import auto_set_test_seed
import unittest
from itertools import product

from spflow.meta.dispatch import init_default_sampling_context
from spflow import log_likelihood, sample, marginalize
from tests.utils.leaves import make_normal_leaf, make_normal_data
from spflow.learn import expectation_maximization
from spflow.modules import Product
import pytest
import torch

# Constants
in_channels_values = [1, 3]
out_features_values = [1, 4]
params = list(product(in_channels_values, out_features_values))


def make_product(in_channels=None, out_features=None, inputs=None):
    if inputs is None:
        inputs = make_normal_leaf(out_features=out_features, out_channels=in_channels)
    return Product(inputs=inputs)


@pytest.mark.parametrize("in_channels,out_features", params)
def test_log_likelihood(in_channels: int, out_features: int):
    product_layer = make_product(in_channels=in_channels, out_features=out_features)
    data = make_normal_data(out_features=out_features)
    lls = log_likelihood(product_layer, data)
    assert lls.shape == (data.shape[0], 1, product_layer.out_channels)


@pytest.mark.parametrize("in_channels,out_features", params)
def test_sample(in_channels: int, out_features: int):
    n_samples = 10
    product_layer = make_product(in_channels=in_channels, out_features=out_features)
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)
    for i in range(product_layer.out_channels):
        data = torch.full((n_samples, out_features), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, out_features), fill_value=i)
        samples = sample(product_layer, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, product_layer.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("in_channels,out_features", params)
def test_expectation_maximization(in_channels: int, out_features: int):
    product_layer = make_product(in_channels=in_channels, out_features=out_features)
    data = make_normal_data(out_features=out_features)
    expectation_maximization(product_layer, data, max_steps=10)


def test_constructor():
    pass


@pytest.mark.parametrize(
    "prune,in_channels,marg_rvs",
    product(
        [True, False],
        in_channels_values,
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
    ),
)
def test_marginalize(prune, in_channels: int, marg_rvs: list[int]):
    out_features = 3
    module = make_product(in_channels=in_channels, out_features=out_features)

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


if __name__ == "__main__":
    unittest.main()
