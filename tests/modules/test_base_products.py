from itertools import product

import pytest
import torch

from spflow.exceptions import ScopeError
from spflow.learn import expectation_maximization
from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.products import ElementwiseProduct
from spflow.modules.products.outer_product import OuterProduct
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_data, make_leaf

cls_values = [ElementwiseProduct, OuterProduct]
in_channels_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]
num_repetitions = [1, 7]
params = list(product(in_channels_values, out_channels_values, out_features_values, num_repetitions))


def make_module(cls, out_features: int, in_channels: int, scopes=None, num_repetitions=None):
    if scopes is None:
        scope_a = Scope(list(range(out_features)))
        scope_b = Scope(list(range(out_features, out_features * 2)))
        scope_c = Scope(list(range(out_features * 2, out_features * 3)))
    else:
        scope_a, scope_b, scope_c = scopes
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_a, num_repetitions=num_repetitions)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_b, num_repetitions=num_repetitions)
    inputs_c = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_c, num_repetitions=num_repetitions)
    inputs = [inputs_a, inputs_b, inputs_c]

    return cls(inputs=inputs)


@pytest.mark.parametrize(
    "cls,in_channels,out_features, num_reps",
    product(cls_values, in_channels_values, [1, 6], num_repetitions),
)
def test_log_likelihood(cls, in_channels: int, out_features: int, num_reps):
    module = make_module(
        cls=cls, out_features=out_features, in_channels=in_channels, num_repetitions=num_reps
    )
    data = make_data(cls=Normal, out_features=out_features * len(module.inputs))
    lls = module.log_likelihood(data)
    # Always expect 4D output [batch, features, channels, num_reps]
    assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)

@pytest.mark.parametrize(
    "cls,out_features,num_reps", product(cls_values, out_features_values, num_repetitions)
)
def test_log_likelihood_broadcasting_channels(cls, out_features: int, num_reps):
    # Define the scopes
    in_channels_a = 1
    in_channels_b = 3
    in_channels_c = 3
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))
    scope_c = Scope(list(range(out_features * 2, out_features * 3)))

    # Define the inputs
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels_a, scope=scope_a, num_repetitions=num_reps)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels_b, scope=scope_b, num_repetitions=num_reps)
    inputs_c = make_leaf(cls=Normal, out_channels=in_channels_c, scope=scope_c, num_repetitions=num_reps)
    inputs = [inputs_a, inputs_b, inputs_c]

    # Create the module
    module = cls(inputs=inputs)

    # Create the data
    data = make_data(cls=Normal, out_features=out_features * len(module.inputs))

    # Compute the log-likelihood
    lls = module.log_likelihood(data)
    # Always expect 4D output [batch, features, channels, num_reps]
    assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)


@pytest.mark.parametrize(
    "cls,in_channels,out_features, num_reps",
    product(cls_values, in_channels_values, [1, 6], num_repetitions),
)
def test_sample(cls, in_channels: int, out_features: int, num_reps):
    n_samples = 10000
    module = make_module(
        cls=cls, out_features=out_features, in_channels=in_channels, num_repetitions=num_reps
    )

    data = torch.full((n_samples, out_features * len(module.inputs)), torch.nan)
    mask = torch.full((n_samples, module.out_features), True, dtype=torch.bool)
    channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
    # Always set repetition_index since num_reps is never None
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = module.sample(data=data, sampling_ctx=sampling_ctx)

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "cls,out_features,num_reps", product(cls_values, out_features_values, num_repetitions)
)
def test_sample_two_inputs_broadcasting_channels(cls, out_features: int, num_reps):
    # Define the scopes
    in_channels_a = 1
    in_channels_b = 3
    in_channels_c = 3
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))
    scope_c = Scope(list(range(out_features * 2, out_features * 3)))

    # Define the inputs
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels_a, scope=scope_a, num_repetitions=num_reps)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels_b, scope=scope_b, num_repetitions=num_reps)
    inputs_c = make_leaf(cls=Normal, out_channels=in_channels_c, scope=scope_c, num_repetitions=num_reps)
    inputs = [inputs_a, inputs_b, inputs_c]

    # Create the module
    module = cls(inputs=inputs)

    n_samples = 5
    data = torch.full((n_samples, out_features * len(module.inputs)), torch.nan)
    channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
    mask = torch.full((n_samples, module.out_features), True, dtype=torch.bool)
    # Always set repetition_index since num_reps is never None
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = module.sample(data=data, sampling_ctx=sampling_ctx)

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "cls,in_channels,out_features,num_reps",
    product(cls_values, in_channels_values, out_features_values, num_repetitions),
)
def test_scopes(cls, in_channels: int, out_features: int, num_reps):
    module = make_module(
        cls=cls, out_features=out_features, in_channels=in_channels, num_repetitions=num_reps
    )
    assert module.scope.query == tuple(range(out_features * len(module.inputs)))


@pytest.mark.parametrize(
    "cls,in_channels,out_features,num_reps",
    product(cls_values, in_channels_values, [2, 6], num_repetitions),
)
def test_expectation_maximization(cls, in_channels: int, out_features: int, num_reps):
    module = make_module(
        cls=cls, out_features=out_features, in_channels=in_channels, num_repetitions=num_reps
    )
    data = make_data(cls=Normal, out_features=out_features * len(module.inputs))
    expectation_maximization(module, data, max_steps=2)


@pytest.mark.parametrize(
    "cls,in_channels,out_features,num_reps",
    product(cls_values, in_channels_values, out_features_values, num_repetitions),
)
def test_invalid_non_disjoint_scopes(cls, in_channels: int, out_features: int, num_reps):
    with pytest.raises(ScopeError):
        make_module(
            cls=cls,
            out_features=out_features,
            in_channels=in_channels,
            scopes=(Scope(range(out_features)), Scope(range(out_features)), Scope(range(out_features))),
            num_repetitions=num_reps,
        )


# @pytest.mark.parametrize("cls", cls_values)
# def test_invalid_out_features_number(cls):
#     with pytest.raises(ValueError):
#         cls(
#             inputs=[
#                 make_normal_leaf(scope=Scope([0]), out_channels=3),
#                 make_normal_leaf(scope=Scope([1, 2, 3]), out_channels=3),
#                 make_normal_leaf(scope=Scope([4, 5, 6]), out_channels=3),
#             ],
#         )

test_cases = [
    ([(3, 4), (3, 4)], True, ElementwiseProduct),
    ([(3, 4), (3, 1)], True, ElementwiseProduct),
    ([(3, 4), (1, 4)], True, ElementwiseProduct),
    ([(3, 1), (1, 4)], True, ElementwiseProduct),
    ([(3, 4), (4, 3)], False, ElementwiseProduct),
    ([(3, 4), (3, 5)], True, OuterProduct),
    ([(3, 4), (3, 1)], True, OuterProduct),
    ([(3, 4), (1, 5)], True, OuterProduct),
    ([(3, 4), (4, 3)], False, OuterProduct),
]


@pytest.mark.parametrize("shape, label, product", test_cases)
def test_broadcast(shape, label, product):
    leaf_layer = []
    current_num_features = 0
    for s in shape:
        out_features, out_channels = s
        scope = Scope(list(range(current_num_features, out_features + current_num_features)))
        leaf_layer.append(make_normal_leaf(scope=scope, out_features=out_features, out_channels=out_channels))
        current_num_features += out_features

    if not label:
        with pytest.raises(ValueError):
            product(inputs=leaf_layer)
    else:
        prod = product(inputs=leaf_layer)
        assert prod is not None


# @pytest.mark.parametrize("cls,prune", product(cls_values, [True, False]))
# def test_marginalize_single_input(cls, prune: bool, split_cls):
#     out_features = 6
#     out_channels = 6
#     leaf_layer = make_normal_leaf(out_features=out_features, out_channels=out_channels)
#     split = SplitHalves(inputs=leaf_layer)
#     module = cls(inputs=[split])
