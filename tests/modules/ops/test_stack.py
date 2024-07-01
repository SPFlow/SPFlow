from tests.fixtures import auto_set_test_seed
import unittest

from itertools import product
from spflow.meta.data import Scope
import pytest
from spflow.meta.dispatch import init_default_sampling_context
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import Sum
from spflow.modules.leaf import Categorical, Binomial
from spflow.modules.ops.stack import Stack
from tests.utils.leaves import make_normal_leaf, make_normal_data
import torch


out_channels_values = [1, 5]
out_features_values = [1, 6]
params = list(product(out_channels_values, out_features_values))


def make_stack(out_channels=3, out_features=3):
    # different scopes
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels)

    return Stack(inputs=[inputs_a, inputs_b])


@pytest.mark.parametrize("out_channels,out_features", params)
def test_log_likelihood(out_channels: int, out_features: int):
    out_channels = 3
    module = make_stack(
        out_channels=out_channels,
        out_features=out_features,
    )
    data = make_normal_data(out_features=module.out_features)
    lls = log_likelihood(module, data)
    assert lls.shape == (data.shape[0], module.out_features, module.out_channels, 2)


@pytest.mark.parametrize("out_channels,out_features", params)
def test_sample(out_channels: int, out_features: int):
    n_samples = 10
    module = make_stack(
        out_channels=out_channels,
        out_features=out_features,
    )
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)
    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan)
        sampling_ctx.output_ids = torch.randint(low=0, high=2, size=(n_samples, module.out_features))
        samples = sample(module, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("out_channels,out_features", params)
def test_expectation_maximization(out_channels: int, out_features: int):
    module = make_stack(
        out_channels=out_channels,
        out_features=out_features,
    )
    data = make_normal_data(out_features=module.out_features)
    expectation_maximization(module, data, max_steps=10)


@pytest.mark.parametrize("out_channels,out_features", params)
def test_gradient_descent_optimization(out_channels: int, out_features: int):
    module = make_stack(
        out_channels=out_channels,
        out_features=out_features,
    )
    data = make_normal_data(out_features=module.out_features)

    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    train_gradient_descent(module, data_loader, epochs=10)


def test_invalid_constructor_same_scope():
    out_features = 3
    out_channels = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels)
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels)

    with pytest.raises(ValueError):
        Stack(inputs=[inputs_a, inputs_b])


def test_invalid_constructor_different_channels():
    out_features = 3
    out_channels = 3

    inputs_a = make_normal_leaf(out_features=out_features, out_channels=out_channels)
    inputs_b = make_normal_leaf(out_features=out_features, out_channels=out_channels + 1)

    with pytest.raises(ValueError):
        Stack(inputs=[inputs_a, inputs_b])


def test_invalid_constructor_different_features():
    out_features = 3
    out_channels = 3

    inputs_a = make_normal_leaf(out_features=out_features, out_channels=out_channels)
    inputs_b = make_normal_leaf(out_features=out_features + 1, out_channels=out_channels)

    with pytest.raises(ValueError):
        Stack(inputs=[inputs_a, inputs_b])


@pytest.mark.parametrize(
    "prune,out_channels,marg_rvs",
    product(
        [True, False],
        out_channels_values,
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
    ),
)
def test_marginalize(prune, out_channels: int, marg_rvs: list[int]):
    out_features = 4
    module = make_stack(
        out_channels=out_channels,
        out_features=out_features,
    )

    # Marginalize scope
    marginalized_module = marginalize(module, marg_rvs, prune=prune)

    if len(marg_rvs) == module.out_features:
        assert marginalized_module is None
        return

    if len(marg_rvs) == module.out_features:
        assert marginalized_module is None
        return
    else:
        assert marginalized_module.out_features == module.out_features - len(marg_rvs)

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


if __name__ == "__main__":
    unittest.main()
