from spflow.modules.leaf import Normal
from tests.fixtures import auto_set_test_seed, auto_set_test_device, auto_set_test_device
import unittest

from itertools import product

from spflow import InvalidParameterCombinationError, ScopeError
from spflow.meta import Scope
import pytest
from spflow.meta import SamplingContext
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import Sum, ElementwiseProduct
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_leaf
import torch

in_channels_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]
num_repetitions = [None, 7]
params = list(product(in_channels_values, out_channels_values, out_features_values, num_repetitions))


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

    return Sum(out_channels=out_channels, inputs=inputs, weights=weights, num_repetitions=num_repetitions)


@pytest.mark.parametrize("in_channels,out_channels,out_features, num_reps", params)
def test_log_likelihood(in_channels: int, out_channels: int, out_features: int, num_reps):
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    data = make_normal_data(out_features=out_features)
    lls = module.log_likelihood(data)
    if num_reps is not None:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)
    else:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels)


# ToDo: Look at this after elementwise product is tested
@pytest.mark.parametrize(
    "in_channels,out_channels,out_features,num_reps",
    product(in_channels_values, out_channels_values, [2, 4], num_repetitions),
)
def test_log_likelihood_product_inputs(in_channels: int, out_channels: int, out_features: int, num_reps):
    inputs_a = make_leaf(
        cls=Normal,
        out_channels=in_channels,
        scope=Scope(range(0, out_features // 2)),
        num_repetitions=num_reps,
    )
    inputs_b = make_leaf(
        cls=Normal,
        out_channels=in_channels,
        scope=Scope(range(out_features // 2, out_features)),
        num_repetitions=num_reps,
    )
    inputs = [inputs_a, inputs_b]
    prod = ElementwiseProduct(inputs=inputs)

    module = Sum(out_channels=out_channels, inputs=prod, num_repetitions=num_reps)

    data = make_normal_data(out_features=out_features)
    lls = module.log_likelihood(data)
    if num_reps is not None:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)
    else:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_sample(in_channels: int, out_channels: int, out_features: int, num_reps):
    n_samples = 100
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
        mask = torch.full((n_samples, module.out_features), True)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )
        samples = module.sample(data=data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "in_channels,out_channels,out_features, num_reps",
    product(in_channels_values, out_channels_values, [2, 8], num_repetitions),
)
def test_sample_product_inputs(in_channels: int, out_channels: int, out_features: int, num_reps):
    n_samples = 100
    inputs_a = make_leaf(
        cls=Normal,
        out_channels=in_channels,
        scope=Scope(range(0, out_features // 2)),
        num_repetitions=num_reps,
    )
    inputs_b = make_leaf(
        cls=Normal,
        out_channels=in_channels,
        scope=Scope(range(out_features // 2, out_features)),
        num_repetitions=num_reps,
    )
    inputs = [inputs_a, inputs_b]
    prod = ElementwiseProduct(inputs=inputs)

    module = Sum(out_channels=out_channels, inputs=prod, num_repetitions=num_reps)

    for i in range(module.out_channels):
        data = torch.full((n_samples, out_features), torch.nan)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
        mask = torch.full((n_samples, module.out_features), True)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )
        samples = module.sample(data=data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "in_channels,out_channels, num_reps",
    list(product(in_channels_values, out_channels_values, num_repetitions)),
)
def test_conditional_sample(in_channels: int, out_channels: int, num_reps):
    out_features = 6
    n_samples = 100
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )

    for i in range(module.out_channels):
        # Create some data
        data = torch.randn(n_samples, module.out_features)

        # Set first three scopes to nan
        data[:, [0, 1, 2]] = torch.nan

        data_copy = data.clone()

        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
        mask = torch.full(channel_index.shape, True)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )

        cache = {}
        samples = module.sample_with_evidence(
            evidence=data,
            is_mpe=False,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

        # Check that log_likelihood is cached
        cached_ll = cache["log_likelihood"][module.inputs]
        assert cached_ll is not None
        assert cached_ll.isfinite().all()

        # Check for correct shape
        assert samples.shape == data.shape

        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()

        # Check, that the last three scopes (those that were conditioned on) are still the same
        assert torch.allclose(data_copy[:, [3, 4, 5]], samples[:, [3, 4, 5]])


@pytest.mark.parametrize("in_channels,out_channels,out_features, num_reps", params)
def test_expectation_maximization(in_channels: int, out_channels: int, out_features: int, num_reps):
    out_channels = 6
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    data = make_normal_data(out_features=out_features)
    expectation_maximization(module, data, max_steps=10)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_gradient_descent_optimization(in_channels: int, out_channels: int, out_features: int, num_reps):
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    data = make_normal_data(out_features=out_features, num_samples=20)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    # Store weights before
    weights_before = module.weights.clone()

    # Run optimization
    train_gradient_descent(module, data_loader, epochs=1)

    # Check that weights have changed
    if in_channels > 1:  # If in_channels is 1, the weight is 1.0 anyway
        assert not torch.allclose(module.weights, weights_before)


@pytest.mark.parametrize("in_channels,out_channels,out_features, num_reps", params)
def test_weights(in_channels: int, out_channels: int, out_features: int, num_reps):
    if num_reps is not None:
        weights = torch.ones((out_features, in_channels, out_channels, num_reps))
    else:
        weights = torch.ones((out_features, in_channels, out_channels))
    weights = weights / weights.sum(dim=1, keepdim=True)

    module = make_sum(
        weights=weights,
        in_channels=in_channels,
        num_repetitions=num_reps,
    )
    assert torch.allclose(module.weights.sum(dim=module.sum_dim), torch.tensor(1.0))
    assert torch.allclose(module.log_weights, module.weights.log())


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_invalid_weights_normalized(in_channels: int, out_channels: int, out_features: int, num_reps):
    if num_reps is not None:
        weights = torch.rand((out_features, in_channels, out_channels, num_reps))
    else:
        weights = torch.rand((out_features, in_channels, out_channels))
    with pytest.raises(ValueError):
        make_sum(weights=weights, in_channels=in_channels, num_repetitions=num_reps)


@pytest.mark.parametrize("in_channels,out_channels,out_features, num_reps", params)
def test_invalid_weights_negative(in_channels: int, out_channels: int, out_features: int, num_reps):
    if num_reps is not None:
        weights = torch.rand((out_features, in_channels, out_channels, num_reps)) - 1.0
    else:
        weights = torch.rand((out_features, in_channels, out_channels)) - 1.0
    with pytest.raises(ValueError):
        make_sum(weights=weights, in_channels=in_channels, num_repetitions=num_reps)


@pytest.mark.parametrize("in_channels,out_channels,out_features, num_reps", params)
def test_invalid_specification_of_out_channels_and_weights(
    in_channels: int, out_channels: int, out_features: int, num_reps
):
    with pytest.raises(ValueError):
        if num_reps is not None:
            weights = torch.rand((out_features, in_channels, out_channels, num_reps))
        else:
            weights = torch.rand((out_features, in_channels, out_channels))
        weights = weights / weights.sum(dim=2, keepdim=True)
        Sum(
            weights=weights,
            inputs=make_normal_leaf(out_features=out_features, out_channels=3, num_repetitions=num_reps),
            num_repetitions=num_reps,
        )


@pytest.mark.parametrize("in_channels,out_channels,out_features, num_reps", params)
def test_invalid_parameter_combination(in_channels: int, out_channels: int, out_features: int, num_reps):
    if num_reps is not None:
        weights = torch.rand((out_features, in_channels, out_channels, num_reps)) + 1.0
    else:
        weights = torch.rand((out_features, in_channels, out_channels)) + 1.0
    with pytest.raises(InvalidParameterCombinationError):
        make_sum(
            weights=weights, out_channels=out_channels, in_channels=in_channels, num_repetitions=num_reps
        )


@pytest.mark.parametrize(
    "prune,in_channels,out_channels,marg_rvs, num_reps",
    product(
        [True, False],
        in_channels_values,
        out_channels_values,
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        num_repetitions,
    ),
)
def test_marginalize(prune, in_channels: int, out_channels: int, marg_rvs: list[int], num_reps):
    out_features = 3
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    weights_shape = module.weights.shape

    # Marginalize scope
    marginalized_module = module.marginalize(marg_rvs, prune=prune)

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


def test_multiple_input():
    in_channels = 2
    out_channels = 2
    out_features = 4
    num_reps = 5
    sum_out_channels = 3

    mean = torch.rand((out_features, out_channels, num_reps))
    std = torch.rand((out_features, out_channels, num_reps))

    normal_layer_a = make_normal_leaf(
        out_features=out_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
        mean=mean,
        std=std,
    )
    normal_layer_b1 = make_normal_leaf(
        out_features=out_features,
        out_channels=1,
        num_repetitions=num_repetitions,
        mean=mean[:, 0:1, :],
        std=std[:, 0:1, :],
    )
    normal_layer_b2 = make_normal_leaf(
        out_features=out_features,
        out_channels=1,
        num_repetitions=num_repetitions,
        mean=mean[:, 1:2, :],
        std=std[:, 1:2, :],
    )

    module_a = Sum(inputs=normal_layer_a, out_channels=sum_out_channels, num_repetitions=num_reps)

    module_b = Sum(
        inputs=[normal_layer_b1, normal_layer_b2], weights=module_a.weights, num_repetitions=num_reps
    )

    # test log likelihood

    data = make_normal_data(out_features=out_features)

    ll_a = module_a.log_likelihood(data)
    ll_b = module_b.log_likelihood(data)

    assert torch.allclose(ll_a, ll_b)

    # test sampling

    n_samples = 10

    data_a = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(low=0, high=sum_out_channels, size=(n_samples, out_features))
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
