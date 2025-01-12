from spflow.modules import ElementwiseProduct
from tests.fixtures import auto_set_test_seed
from itertools import product

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
import unittest

import pytest
from spflow.meta.dispatch import init_default_sampling_context, SamplingContext
from spflow.meta.data import Scope
from spflow.modules.outer_product import OuterProduct
from spflow.modules.factorize import Factorize
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_data, make_leaf
from spflow.modules.leaf import Normal
import torch

cls_values = [OuterProduct, ElementwiseProduct]
in_channels_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]
params = list(product(in_channels_values, out_channels_values, out_features_values))


def make_module(cls, out_features: int, in_channels: int, scopes=None):
    if scopes is None:
        scope_a = Scope(list(range(out_features)))
        scope_b = Scope(list(range(out_features, out_features * 2)))
        scope_c = Scope(list(range(out_features * 2, out_features * 3)))
    else:
        scope_a, scope_b, scope_c = scopes
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_a)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_b)
    inputs_c = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_c)
    inputs = [inputs_a, inputs_b, inputs_c]

    return cls(inputs=inputs)


@pytest.mark.parametrize(
    "cls,in_channels,out_features",
    product(cls_values, in_channels_values, [1, 6]),
)
def test_log_likelihood(cls, in_channels: int, out_features: int):
    module = make_module(cls=cls, out_features=out_features, in_channels=in_channels)
    data = make_data(cls=Normal, out_features=out_features * len(module.inputs))
    lls = log_likelihood(module, data)
    assert lls.shape == (data.shape[0], module.out_features, module.out_channels)


@pytest.mark.parametrize("cls,out_features", product(cls_values, out_features_values))
def test_log_likelihood_broadcasting_channels(cls, out_features: int):
    # Define the scopes
    in_channels_a = 1
    in_channels_b = 3
    in_channels_c = 3
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))
    scope_c = Scope(list(range(out_features * 2, out_features * 3)))

    # Define the inputs
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels_a, scope=scope_a)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels_b, scope=scope_b)
    inputs_c = make_leaf(cls=Normal, out_channels=in_channels_c, scope=scope_c)
    inputs = [inputs_a, inputs_b, inputs_c]

    # Create the module
    module = cls(inputs=inputs)

    # Create the data
    data = make_data(cls=Normal, out_features=out_features * len(module.inputs))

    # Compute the log-likelihood
    lls = log_likelihood(module, data)
    assert lls.shape == (data.shape[0], out_features, module.out_channels)


@pytest.mark.parametrize(
    "cls,in_channels,out_features",
    product(cls_values, in_channels_values, [1, 6]),
)
def test_sample(cls, in_channels: int, out_features: int):
    n_samples = 10

    module = make_module(cls=cls, out_features=out_features, in_channels=in_channels)

    data = torch.full((n_samples, out_features * len(module.inputs)), torch.nan)
    mask = torch.full((n_samples, module.out_features), True, dtype=torch.bool)
    channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
    samples = sample(module, data, sampling_ctx=sampling_ctx)

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("cls,out_features", product(cls_values, out_features_values))
def test_sample_two_inputs_broadcasting_channels(cls, out_features: int):
    # Define the scopes
    in_channels_a = 1
    in_channels_b = 3
    in_channels_c = 3
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))
    scope_c = Scope(list(range(out_features * 2, out_features * 3)))

    # Define the inputs
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels_a, scope=scope_a)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels_b, scope=scope_b)
    inputs_c = make_leaf(cls=Normal, out_channels=in_channels_c, scope=scope_c)
    inputs = [inputs_a, inputs_b, inputs_c]

    # Create the module
    module = cls(inputs=inputs)

    n_samples = 5
    data = torch.full((n_samples, out_features * len(module.inputs)), torch.nan)
    channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
    mask = torch.full((n_samples, module.out_features), True, dtype=torch.bool)
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
    samples = sample(module, data, sampling_ctx=sampling_ctx)

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "cls,in_channels,out_features", product(cls_values, in_channels_values, out_features_values)
)
def test_scopes(cls, in_channels: int, out_features: int):
    module = make_module(cls=cls, out_features=out_features, in_channels=in_channels)
    assert module.scope.query == list(range(out_features * len(module.inputs)))


@pytest.mark.parametrize(
    "cls,in_channels,out_features",
    product(
        cls_values,
        in_channels_values,
        [2, 6],
    ),
)
def test_expectation_maximization(cls, in_channels: int, out_features: int):
    module = make_module(cls=cls, out_features=out_features, in_channels=in_channels)
    data = make_data(cls=Normal, out_features=out_features * len(module.inputs))
    expectation_maximization(module, data, max_steps=2)


@pytest.mark.parametrize(
    "cls,in_channels,out_features",
    product(cls_values, in_channels_values, out_features_values),
)
def test_invalid_non_disjoint_scopes(cls, in_channels: int, out_features: int):
    with pytest.raises(ScopeError):
        make_module(
            cls=cls,
            out_features=out_features,
            in_channels=in_channels,
            scopes=(Scope(range(out_features)), Scope(range(out_features)), Scope(range(out_features))),
        )


@pytest.mark.parametrize("cls", cls_values)
def test_invalid_out_features_number(cls):
    with pytest.raises(ValueError):
        cls(
            inputs=[
                make_normal_leaf(scope=Scope([0]), out_channels=3),
                make_normal_leaf(scope=Scope([1, 2, 3]), out_channels=3),
                make_normal_leaf(scope=Scope([4, 5, 6]), out_channels=3),
            ],
        )


@pytest.mark.parametrize("cls,prune", product(cls_values, [True, False]))
def test_marginalize_single_input(cls, prune: bool):
    # TODO: implement marginalization
    pass
