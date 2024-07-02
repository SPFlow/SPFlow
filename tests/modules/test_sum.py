from tests.fixtures import auto_set_test_seed
import unittest

from itertools import product

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.data import Scope
import pytest
from spflow.meta.dispatch import init_default_sampling_context, init_default_dispatch_context
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import Sum
from tests.utils.leaves import make_normal_leaf, make_normal_data
import torch

in_channels_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]
is_single_input_values = [True, False]
params = list(product(in_channels_values, out_channels_values, out_features_values, is_single_input_values))


def make_sum(in_channels=None, out_channels=None, out_features=None, weights=None, is_single_input=True):
    if isinstance(weights, list):
        weights = torch.tensor(weights)
        if weights.dim() == 1:
            weights = weights.unsqueeze(1).unsqueeze(2)
        elif weights.dim() == 2:
            weights = weights.unsqueeze(2)

    if weights is not None:
        out_features = weights.shape[0]

    if is_single_input:
        inputs = make_normal_leaf(out_features=out_features, out_channels=in_channels)
    else:
        inputs_a = make_normal_leaf(out_features=out_features, out_channels=in_channels)
        inputs_b = make_normal_leaf(out_features=out_features, out_channels=in_channels)
        inputs = [inputs_a, inputs_b]

    return Sum(out_channels=out_channels, inputs=inputs, weights=weights)


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_log_likelihood(in_channels: int, out_channels: int, out_features: int, is_single_input: bool):
    out_channels = 3
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        is_single_input=is_single_input,
    )
    data = make_normal_data(out_features=out_features)
    lls = log_likelihood(module, data)
    assert lls.shape == (data.shape[0], module.out_features, module.out_channels)


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_sample(in_channels: int, out_channels: int, out_features: int, is_single_input: bool):
    n_samples = 100
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        is_single_input=is_single_input,
    )
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)
    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan)
        sampling_ctx.output_ids = torch.randint(
            low=0, high=module.out_channels, size=(n_samples, module.out_features)
        )
        samples = sample(module, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "in_channels,out_channels,is_single_input",
    list(product(in_channels_values, out_channels_values, is_single_input_values)),
)
def test_conditional_sample(in_channels: int, out_channels: int, is_single_input: bool):
    out_features = 6
    n_samples = 100
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        is_single_input=is_single_input,
    )
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    for i in range(module.out_channels):
        # Create some data
        data = torch.randn(n_samples, module.out_features)

        # Set first three scopes to nan
        data[:, [0, 1, 2]] = torch.nan

        data_copy = data.clone()

        # Perform log-likelihood computation
        dispatch_ctx = init_default_dispatch_context()
        _ = log_likelihood(module, data, dispatch_ctx=dispatch_ctx)

        # Check that log_likelihood is cached
        assert dispatch_ctx.cache["log_likelihood"][module] is not None
        assert dispatch_ctx.cache["log_likelihood"][module].isfinite().all()

        sampling_ctx.output_ids = torch.randint(
            low=0, high=module.out_channels, size=(n_samples, module.out_features)
        )
        samples = sample(module, data, sampling_ctx=sampling_ctx, dispatch_ctx=dispatch_ctx)
        assert samples.shape == data.shape

        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()

        # Check, that the last three scopes (those that were conditioned on) are still the same
        assert torch.allclose(data_copy[:, [3, 4, 5]], samples[:, [3, 4, 5]])


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_expectation_maximization(
    in_channels: int, out_channels: int, out_features: int, is_single_input: bool
):
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        is_single_input=is_single_input,
    )
    data = make_normal_data(out_features=out_features)
    expectation_maximization(module, data, max_steps=10)


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_gradient_descent_optimization(
    in_channels: int, out_channels: int, out_features: int, is_single_input: bool
):
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        is_single_input=is_single_input,
    )
    data = make_normal_data(out_features=out_features, num_samples=20)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    train_gradient_descent(module, data_loader, epochs=1)


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_weights(in_channels: int, out_channels: int, out_features: int, is_single_input: bool):
    if is_single_input:
        weights = torch.ones((out_features, in_channels, out_channels))
        weights = weights / weights.sum(dim=1, keepdim=True)
    else:
        weights = torch.ones((out_features, in_channels, out_channels, 2))
        weights = weights / weights.sum(dim=3, keepdim=True)

    module = make_sum(
        weights=weights,
        in_channels=in_channels,
        is_single_input=is_single_input,
    )
    assert torch.allclose(module.weights.sum(dim=module.sum_dim), torch.tensor(1.0))
    assert torch.allclose(module.log_weights, module.weights.log())


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_invalid_weights_normalized(
    in_channels: int, out_channels: int, out_features: int, is_single_input: bool
):
    weights = torch.rand((out_features, in_channels, out_channels))
    with pytest.raises(ValueError):
        make_sum(weights=weights, in_channels=in_channels)


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_invalid_weights_negative(
    in_channels: int, out_channels: int, out_features: int, is_single_input: bool
):
    weights = torch.rand((out_features, in_channels, out_channels)) - 1.0
    with pytest.raises(ValueError):
        make_sum(weights=weights, in_channels=in_channels)


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_invalid_input_channels_mismatch(
    in_channels: int, out_channels: int, out_features: int, is_single_input: bool
):
    input_a = make_normal_leaf(out_features=out_features, out_channels=in_channels + 1)
    input_b = make_normal_leaf(out_features=out_features, out_channels=in_channels + 2)
    with pytest.raises(ValueError):
        Sum(out_channels=out_channels, inputs=[input_a, input_b])


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_invalid_input_features_mismatch(
    in_channels: int, out_channels: int, out_features: int, is_single_input: bool
):
    input_a = make_normal_leaf(out_features=out_features + 1, out_channels=in_channels)
    input_b = make_normal_leaf(out_features=out_features + 2, out_channels=in_channels)
    with pytest.raises(ValueError):
        Sum(out_channels=out_channels, inputs=[input_a, input_b])


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_invalid_specification_of_out_channels_and_weights(
    in_channels: int, out_channels: int, out_features: int, is_single_input: bool
):
    with pytest.raises(ValueError):
        weights = torch.rand((out_features, in_channels, out_channels))
        weights = weights / weights.sum(dim=2, keepdim=True)
        Sum(weights=weights, inputs=make_normal_leaf(out_features=out_features, out_channels=3))


@pytest.mark.parametrize("in_channels,out_channels,out_features,is_single_input", params)
def test_invalid_parameter_combination(
    in_channels: int, out_channels: int, out_features: int, is_single_input: bool
):
    weights = torch.rand((out_features, in_channels, out_channels)) + 1.0
    with pytest.raises(InvalidParameterCombinationError):
        make_sum(weights=weights, out_channels=out_channels, in_channels=in_channels)


@pytest.mark.parametrize(
    "in_channels,out_channels,out_features",
    product(in_channels_values, out_channels_values, out_features_values),
)
def test_same_scope_error(in_channels: int, out_channels: int, out_features: int):
    with pytest.raises(ScopeError):
        input_a = make_normal_leaf(scope=Scope(range(0, out_features)), out_channels=in_channels)
        input_b = make_normal_leaf(
            scope=Scope(range(out_features, out_features * 2)), out_channels=in_channels
        )
        with pytest.raises(ValueError):
            Sum(out_channels=out_channels, inputs=[input_a, input_b])


@pytest.mark.parametrize(
    "prune,in_channels,out_channels,is_single_input,marg_rvs",
    product(
        [True, False],
        in_channels_values,
        out_channels_values,
        is_single_input_values,
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
    ),
)
def test_marginalize(prune, in_channels: int, out_channels: int, is_single_input: bool, marg_rvs: list[int]):
    out_features = 3
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        is_single_input=is_single_input,
    )
    weights_shape = module.weights.shape

    # Marginalize scope
    marginalized_module = marginalize(module, marg_rvs, prune=prune)

    if len(marg_rvs) == out_features:
        assert marginalized_module is None
        return

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0

    # Weights num_scopes dimension should be reduced by len(marg_rv)
    assert marginalized_module.weights.shape[0] == weights_shape[0] - len(marg_rvs)

    # Check that all other dims stayed the same
    for d in range(1, len(weights_shape)):
        assert marginalized_module.weights.shape[d] == weights_shape[d]


if __name__ == "__main__":
    unittest.main()
