from itertools import product

import pytest
import torch

from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.meta import SamplingContext
from spflow.meta import Scope
from spflow.modules.leaf import Categorical, Binomial
from spflow.modules.ops import Cat
from tests.utils.leaves import make_normal_leaf, make_normal_data

out_channels_values = [1, 5]
out_features_values = [1, 6]
dim_values = [1, 2]
num_repetitions = [None, 5]
params = list(product(out_channels_values, out_features_values, num_repetitions, dim_values))


def make_cat(out_channels=3, out_features=3, num_repetitions=None, dim=1):
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
def test_log_likelihood(out_channels: int, out_features: int, num_reps, dim: int, device):
    out_channels = 3
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    ).to(device)
    data = make_normal_data(out_features=module.out_features).to(device)
    lls = module.log_likelihood(data)
    if num_reps == None:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels)
    else:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_sample(out_channels: int, out_features: int, num_reps, dim: int, device):
    n_samples = 10
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    ).to(device)
    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan).to(device)
        channel_index = torch.randint(
            low=0, high=module.out_channels, size=(n_samples, module.out_features)
        ).to(device)
        mask = torch.full((n_samples, module.out_features), True, dtype=torch.bool).to(device)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,)).to(device)
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )
        samples = module.sample(data=data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_expectation_maximization(out_channels: int, out_features: int, num_reps, dim: int, device):
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    ).to(device)
    data = make_normal_data(out_features=module.out_features).to(device)
    expectation_maximization(module, data, max_steps=10)


@pytest.mark.parametrize("out_channels,out_features, num_reps, dim", params)
def test_gradient_descent_optimization(out_channels: int, out_features: int, num_reps, dim: int, device):
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    ).to(device)
    data = make_normal_data(out_features=module.out_features).to(device)

    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    train_gradient_descent(module, data_loader, epochs=10)


def test_invalid_constructor_same_scope_dim1(device):
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=1).to(device)


def test_invalid_constructor_different_scope_dim2(device):
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=2).to(device)


def test_invalid_constructor_different_channels_dim1(device):
    out_features = 3
    out_channels = 3
    num_repetitions = 3
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, 2 * out_features)))

    inputs_a = make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions).to(
        device
    )
    inputs_b = make_normal_leaf(scope_b, out_channels=out_channels + 1, num_repetitions=num_repetitions).to(
        device
    )

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=1).to(device)


def test_invalid_constructor_different_features_dim2(device):
    out_features = 3
    out_channels = 3
    num_repetitions = 3

    inputs_a = make_normal_leaf(
        out_features=out_features, out_channels=out_channels, num_repetitions=num_repetitions
    ).to(device)
    inputs_b = make_normal_leaf(
        out_features=out_features + 1, out_channels=out_channels, num_repetitions=num_repetitions
    ).to(device)

    with pytest.raises(ValueError):
        Cat(inputs=[inputs_a, inputs_b], dim=2).to(device)


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
def test_marginalize(prune, out_channels: int, dim: int, marg_rvs: list[int], num_reps, device):
    out_features = 4
    module = make_cat(
        out_channels=out_channels,
        out_features=out_features,
        dim=dim,
        num_repetitions=num_reps,
    ).to(device)

    # Marginalize scope
    marginalized_module = module.marginalize(marg_rvs, prune=prune)

    if len(marg_rvs) == module.out_features:
        assert marginalized_module is None
        return
    else:
        assert marginalized_module.out_features == module.out_features - len(marg_rvs)

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


def test_marginalize_one_of_two_inputs(device):
    out_channels = 3
    num_repetitions = 3
    inputs_cat = Categorical(scope=Scope([0, 1]), p=torch.rand(2, out_channels, num_repetitions)).to(device)
    inputs_bin = Binomial(
        scope=Scope([2, 3, 4]),
        n=torch.ones((3, out_channels, num_repetitions)) * 3,
        p=torch.rand(3, out_channels, num_repetitions),
        num_repetitions=num_repetitions,
    ).to(device)

    module = Cat(inputs=[inputs_cat, inputs_bin], dim=1).to(device)

    # Marginalize categorical scope, expect binomial to be returned
    marg_rvs_cat = inputs_cat.scope.query
    marginalized_module = module.marginalize(marg_rvs_cat, prune=True)
    assert isinstance(marginalized_module, type(inputs_bin))

    # Marginalize binomial scope, expect categorical to be returned
    marg_rvs_bin = inputs_bin.scope.query
    marginalized_module = module.marginalize(marg_rvs_bin, prune=True)
    assert isinstance(marginalized_module, type(inputs_cat))
