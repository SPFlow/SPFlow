from tests.fixtures import auto_set_test_seed
import unittest

from itertools import product
from spflow.meta.data import Scope
import pytest
from spflow.meta.dispatch import init_default_sampling_context, SamplingContext
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import Sum
from spflow.modules.leaf import Categorical, Binomial
from spflow.modules.ops.cat import Cat
from tests.utils.leaves import make_normal_leaf, make_normal_data, evaluate_log_likelihood
import torch


out_channels_values = [1, 5]
out_features_values = [1, 6]
dim_values = [1, 2]
num_repetitions = [None, 5]
params = list(product(out_channels_values, out_features_values, num_repetitions, dim_values))


def make_cat(out_channels=3, out_features=3, num_repetitions=None ,dim=1):
    if dim == 1:
        # different scopes
        scope_a = Scope(list(range(0, out_features)))
        scope_b = Scope(list(range(out_features, 2 * out_features)))
    elif dim == 2:
        # Same scopes
        scope_a = Scope(list(range(0, out_features)))
        scope_b = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    return Cat(inputs=[inputs_a, inputs_b], dim=dim)


@pytest.mark.parametrize("out_channels,out_features,num_reps, dim", params)
def test_log_likelihood(out_channels: int, out_features: int, num_reps, dim: int):
    out_channels = 3
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = make_normal_data(out_features=module.out_features)
    lls = log_likelihood(module, data)
    if num_reps == None:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels)
    else:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)



@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_sample(out_channels: int, out_features: int, num_reps, dim: int):
    n_samples = 10
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
        mask = torch.full((n_samples, module.out_features), True, dtype=torch.bool)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
        samples = sample(module, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_expectation_maximization(out_channels: int, out_features: int, num_reps, dim: int):
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = make_normal_data(out_features=module.out_features)
    expectation_maximization(module, data, max_steps=10)


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_gradient_descent_optimization(out_channels: int, out_features: int, num_reps, dim: int):
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = make_normal_data(out_features=module.out_features)

    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    train_gradient_descent(module, data_loader, epochs=10)


def test_invalid_constructor_same_scope_dim1():
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=1)


def test_invalid_constructor_different_scope_dim2():
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=2)


def test_invalid_constructor_different_channels_dim1():
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels + 1, num_repetitions=num_repetitions)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=1)


def test_invalid_constructor_different_features_dim2():
    out_features = 3
    out_channels = 3
    num_repetitions = 3

    inputs_a = make_normal_leaf(out_features=out_features, out_channels=out_channels, num_repetitions=num_repetitions)
    inputs_b = make_normal_leaf(out_features=out_features + 1, out_channels=out_channels, num_repetitions=num_repetitions)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=2)


@pytest.mark.parametrize(
    "prune,out_channels,dim,marg_rvs,num_reps",
    product(
        [True, False],
        out_channels_values,
        dim_values,
        # Create a list of all combinations betweens elements of [{}, 0, 1, 2, 3] and [{}, 0, 1, 2, 3] (where {} means not present
        [
            [0],
            [1],
            [2],
            [3],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
        ],
        num_repetitions,
    ),
)
def test_marginalize(prune, out_channels: int, dim: int, marg_rvs: list[int], num_reps):
    out_features = 4
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        dim=dim,
        num_repetitions=num_reps,
    )

    # Marginalize scope
    marginalized_module = marginalize(module, marg_rvs, prune=prune)

    if len(marg_rvs) == module.out_features:
        assert marginalized_module is None
        return
    else:
        assert marginalized_module.out_features == module.out_features - len(marg_rvs)

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


def test_marginalize_one_of_two_inputs():
    out_channels = 3
    num_repetitions = 3
    inputs_cat = Categorical(scope=Scope([0, 1]), p=torch.rand(2, out_channels, num_repetitions))
    inputs_bin = Binomial(
        scope=Scope([2, 3, 4]),
        n=torch.ones((3, out_channels, num_repetitions)) * 3,
        p=torch.rand(3, out_channels, num_repetitions),
        num_repetitions=num_repetitions,
    )

    module = Cat(inputs=[inputs_cat, inputs_bin], dim=1)

    # Marginalize categorical scope, expect binomial to be returned
    marg_rvs_cat = inputs_cat.scope.query
    marginalized_module = marginalize(module, marg_rvs_cat, prune=True)
    assert isinstance(marginalized_module, type(inputs_bin))

    # Marginalize binomial scope, expect categorical to be returned
    marg_rvs_bin = inputs_bin.scope.query
    marginalized_module = marginalize(module, marg_rvs_bin, prune=True)
    assert isinstance(marginalized_module, type(inputs_cat))
