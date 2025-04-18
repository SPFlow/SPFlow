from spflow.modules.elementwise_sum import ElementwiseSum
from spflow.modules.leaf import Normal
from tests.fixtures import auto_set_test_seed
import unittest

from itertools import product

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.data import Scope
import pytest
from spflow.meta.dispatch import init_default_sampling_context, init_default_dispatch_context, SamplingContext
from spflow import log_likelihood, sample, marginalize, sample_with_evidence
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import ElementwiseProduct
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_leaf
import torch

in_channels_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]
num_repetitions = [None, 7]
params = list(product(in_channels_values, out_channels_values, out_features_values, num_repetitions))


def make_sum(in_channels=None, out_channels=None, out_features=None, weights=None, scopes=None, num_repetitions=None):
    if isinstance(weights, list):
        weights = torch.tensor(weights)
        if weights.dim() == 1:
            weights = weights.unsqueeze(1).unsqueeze(2)
        elif weights.dim() == 2:
            weights = weights.unsqueeze(2)

    if weights is not None:
        out_features = weights.shape[0]

    if scopes is None:
        scope_a = Scope(list(range(out_features)))
        scope_b = Scope(list(range(out_features)))
    else:
        scope_a, scope_b = scopes
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_a, num_repetitions=num_repetitions)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_b, num_repetitions=num_repetitions)
    inputs = [inputs_a, inputs_b]

    return ElementwiseSum(out_channels=out_channels, inputs=inputs, weights=weights, num_repetitions=num_repetitions)


@pytest.mark.parametrize(
    "in_channels,out_channels,out_features,num_reps",
    product(in_channels_values, out_channels_values, out_features_values, num_repetitions)
)
def test_log_likelihood(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps
    ).to(device)
    data = make_normal_data(out_features=out_features).to(device)
    ctx = init_default_dispatch_context()
    lls = log_likelihood(module, data, dispatch_ctx=ctx)
    if num_reps is None:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels)
    else:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)


@pytest.mark.parametrize(
    "in_channels,out_channels,out_features,num_reps", product(in_channels_values, out_channels_values, [2, 4], num_repetitions)
)
def test_log_likelihood_two_product_inputs(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    num_reps = 7
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels, scope=Scope(range(0, out_features // 2)), num_repetitions=num_reps)
    inputs_b = make_leaf(
        cls=Normal, out_channels=in_channels, scope=Scope(range(out_features // 2, out_features), ), num_repetitions=num_reps
    )
    inputs = [inputs_a, inputs_b]
    prod_0 = ElementwiseProduct(inputs=inputs)

    inputs_a = make_leaf(cls=Normal, out_channels=in_channels, scope=Scope(range(0, out_features // 2)), num_repetitions=num_reps)
    inputs_b = make_leaf(
        cls=Normal, out_channels=in_channels, scope=Scope(range(out_features // 2, out_features)), num_repetitions=num_reps
    )
    inputs = [inputs_a, inputs_b]
    prod_1 = ElementwiseProduct(inputs=inputs)

    inputs = [prod_0, prod_1]

    module = ElementwiseSum(out_channels=out_channels, inputs=inputs, num_repetitions=num_reps).to(device)

    data = make_normal_data(out_features=out_features).to(device)
    ctx = init_default_dispatch_context()
    lls = log_likelihood(module, data, dispatch_ctx=ctx)
    if num_reps is None:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels)
    else:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)



@pytest.mark.parametrize("out_channels", out_channels_values)
def test_log_likelihood_broadcasting_channels(out_channels: int, device):
    in_channels_a = 1
    in_channels_b = 3
    out_features = 2
    inputs_a_0 = make_leaf(cls=Normal, out_channels=in_channels_a, scope=Scope(range(0, out_features // 2)))
    inputs_a_1 = make_leaf(
        cls=Normal, out_channels=in_channels_a, scope=Scope(range(out_features // 2, out_features))
    )
    inputs = [inputs_a_0, inputs_a_1]
    prod_0 = ElementwiseProduct(inputs=inputs)

    inputs_b_0 = make_leaf(cls=Normal, out_channels=in_channels_b, scope=Scope(range(0, out_features // 2)))
    inputs_b_1 = make_leaf(
        cls=Normal, out_channels=in_channels_b, scope=Scope(range(out_features // 2, out_features))
    )
    inputs = [inputs_b_0, inputs_b_1]
    prod_1 = ElementwiseProduct(inputs=inputs)
    inputs = [prod_0, prod_1]

    module = ElementwiseSum(out_channels=out_channels, inputs=inputs).to(device)
    data = make_normal_data(out_features=out_features).to(device)
    ctx = init_default_dispatch_context()
    lls = log_likelihood(module, data, dispatch_ctx=ctx)
    assert lls.shape == (data.shape[0], module.out_features, module.out_channels)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_sample(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    n_samples = 100
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps
    ).to(device)


    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan).to(device)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features)).to(device)
        mask = torch.full((n_samples, module.out_features), True).to(device)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,)).to(device)
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
        samples = sample(module, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "in_channels,out_channels,out_features,num_reps", product(in_channels_values, out_channels_values, [2, 8], num_repetitions)
)
def test_sample_two_product_inputs(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    n_samples = 100
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels, scope=Scope(range(0, out_features // 2)), num_repetitions=num_reps)
    inputs_b = make_leaf(
        cls=Normal, out_channels=in_channels, scope=Scope(range(out_features // 2, out_features)), num_repetitions=num_reps
    )
    inputs = [inputs_a, inputs_b]
    prod_0 = ElementwiseProduct(inputs=inputs)

    inputs_a = make_leaf(cls=Normal, out_channels=in_channels, scope=Scope(range(0, out_features // 2)), num_repetitions=num_reps)
    inputs_b = make_leaf(
        cls=Normal, out_channels=in_channels, scope=Scope(range(out_features // 2, out_features)), num_repetitions=num_reps
    )
    inputs = [inputs_a, inputs_b]
    prod_1 = ElementwiseProduct(inputs=inputs)
    inputs = [prod_0, prod_1]

    module = ElementwiseSum(out_channels=out_channels, inputs=inputs, num_repetitions=num_reps).to(device)

    for i in range(module.out_channels):
        data = torch.full((n_samples, out_features), torch.nan).to(device)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features)).to(device)
        mask = torch.full((n_samples, module.out_features), True).to(device)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,)).to(device)
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
        samples = sample(module, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("out_channels", out_channels_values)
def test_sample_two_inputs_broadcasting_channels(out_channels: int, device):
    # Define the scopes
    in_channels_a = 1
    in_channels_b = 3
    out_features = 10

    # Define the inputs
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels_a, out_features=out_features)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels_b, out_features=out_features)
    inputs = [inputs_a, inputs_b]

    # Create the module
    module = ElementwiseSum(inputs=inputs, out_channels=out_channels).to(device)

    n_samples = 5
    data = torch.full((n_samples, out_features), torch.nan).to(device)
    channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features)).to(device)
    mask = torch.full((n_samples, module.out_features), True).to(device)
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
    samples = sample(module, data, sampling_ctx=sampling_ctx)

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "in_channels,out_channels, num_reps",
    list(product(in_channels_values, out_channels_values, num_repetitions)),
)
def test_conditional_sample(in_channels: int, out_channels: int, num_reps, device):
    out_features = 6
    n_samples = 100
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps
    ).to(device)

    for i in range(module.out_channels):
        # Create some data
        data = torch.randn(n_samples, module.out_features).to(device)

        # Set first three scopes to nan
        data[:, [0, 1, 2]] = torch.nan

        data_copy = data.clone()

        # Perform log-likelihood computation
        dispatch_ctx = init_default_dispatch_context()

        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features)).to(device)
        mask = torch.full(channel_index.shape, True).to(device)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,)).to(device)
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)

        samples = sample_with_evidence(
            module,
            data,
            is_mpe=False,
            check_support=False,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=sampling_ctx,
        )

        # Check that log_likelihood is cached
        assert dispatch_ctx.cache["log_likelihood"][module] is not None
        assert dispatch_ctx.cache["log_likelihood"][module].isfinite().all()

        # Check for correct shape
        assert samples.shape == data.shape

        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()

        # Check, that the last three scopes (those that were conditioned on) are still the same
        assert torch.allclose(data_copy[:, [3, 4, 5]], samples[:, [3, 4, 5]])


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_expectation_maximization(
    in_channels: int,
    out_channels: int,
    out_features: int,
    num_reps,
    device
):
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps
    ).to(device)
    data = make_normal_data(out_features=out_features).to(device)
    expectation_maximization(module, data, max_steps=10)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_gradient_descent_optimization(
    in_channels: int,
    out_channels: int,
    out_features: int,
    num_reps,
    device
):
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps
    ).to(device)
    data = make_normal_data(out_features=out_features, num_samples=20).to(device)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    # Store weights before
    weights_before = module.weights.clone()

    # Run optimization
    train_gradient_descent(module, data_loader, epochs=1)

    # Check that weights have changed
    if in_channels > 1:  # If in_channels is 1, the weight is 1.0 anyway
        assert not torch.allclose(module.weights, weights_before)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_weights(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    if num_reps is None:
        weights = torch.ones((out_features, in_channels, out_channels, 2)).to(device)
    else:
        weights = torch.ones((out_features, in_channels, out_channels, 2, num_reps)).to(device)
    weights = weights / weights.sum(dim=3, keepdim=True)

    module = make_sum(
        weights=weights,
        in_channels=in_channels,
        num_repetitions=num_reps,
    ).to(device)
    assert torch.allclose(module.weights.sum(dim=module.sum_dim), torch.tensor(1.0))
    assert torch.allclose(module.log_weights, module.weights.log())


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_invalid_weights_normalized(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    if num_reps is None:
        weights = torch.rand((out_features, in_channels, out_channels)).to(device)
    else:
        weights = torch.rand((out_features, in_channels, out_channels, num_reps)).to(device)
    with pytest.raises(ValueError):
        make_sum(weights=weights, in_channels=in_channels).to(device)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_invalid_weights_negative(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    if num_reps is None:
        weights = torch.rand((out_features, in_channels, out_channels)).to(device) - 1.0
    else:
        weights = torch.rand((out_features, in_channels, out_channels, num_reps)).to(device) - 1.0
    with pytest.raises(ValueError):
        make_sum(weights=weights, in_channels=in_channels).to(device)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_invalid_input_channels_mismatch(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    input_a = make_normal_leaf(out_features=out_features, out_channels=in_channels + 1, num_repetitions=num_reps)
    input_b = make_normal_leaf(out_features=out_features, out_channels=in_channels + 2, num_repetitions=num_reps)
    with pytest.raises(ValueError):
        ElementwiseSum(out_channels=out_channels, inputs=[input_a, input_b]).to(device)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_invalid_input_features_mismatch(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    input_a = make_normal_leaf(out_features=out_features + 1, out_channels=in_channels, num_repetitions=num_reps)
    input_b = make_normal_leaf(out_features=out_features + 2, out_channels=in_channels, num_repetitions=num_reps)
    with pytest.raises(ValueError):
        ElementwiseSum(out_channels=out_channels, inputs=[input_a, input_b]).to(device)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_invalid_specification_of_out_channels_and_weights(
    in_channels: int, out_channels: int, out_features: int, num_reps, device
):
    with pytest.raises(ValueError):
        if num_reps is None:
            weights = torch.rand((out_features, in_channels, out_channels)).to(device)
        else:
            weights = torch.rand((out_features, in_channels, out_channels, num_reps)).to(device)
        weights = weights / weights.sum(dim=2, keepdim=True)
        ElementwiseSum(
            weights=weights,
            inputs=[
                make_normal_leaf(out_features=out_features, out_channels=out_channels + 1, num_repetitions=num_reps),
                make_normal_leaf(out_features=out_features, out_channels=out_channels + 1, num_repetitions=num_reps),
            ],
        ).to(device)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_invalid_parameter_combination(in_channels: int, out_channels: int, out_features: int, num_reps, device):
    if num_reps is None:
        weights = torch.rand((out_features, in_channels, out_channels)).to(device) + 1.0
    else:
        weights = torch.rand((out_features, in_channels, out_channels, num_reps)).to(device) + 1.0
    with pytest.raises(InvalidParameterCombinationError):
        make_sum(weights=weights, out_channels=out_channels, in_channels=in_channels, num_repetitions=num_reps).to(device)


@pytest.mark.parametrize(
    "in_channels,out_channels,out_features",
    product(in_channels_values, out_channels_values, out_features_values),
)
def test_same_scope_error(in_channels: int, out_channels: int, out_features: int, device):
    with pytest.raises(ScopeError):
        input_a = make_normal_leaf(scope=Scope(range(0, out_features)), out_channels=in_channels)
        input_b = make_normal_leaf(
            scope=Scope(range(out_features, out_features * 2)), out_channels=in_channels
        )
        with pytest.raises(ValueError):
            ElementwiseSum(out_channels=out_channels, inputs=[input_a, input_b]).to(device)


@pytest.mark.parametrize(
    "prune,in_channels,out_channels,marg_rvs,num_reps",
    product(
        [True, False],
        in_channels_values,
        out_channels_values,
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        num_repetitions,
    ),
)
def test_marginalize(prune, in_channels: int, out_channels: int, marg_rvs: list[int], num_reps, device):
    out_features = 3
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps
    ).to(device)
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
