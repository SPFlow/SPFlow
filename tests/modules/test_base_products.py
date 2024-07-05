from spflow.modules import ElementwiseProduct
from tests.fixtures import auto_set_test_seed
from itertools import product

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
import unittest

import pytest
from spflow.meta.dispatch import init_default_sampling_context
from spflow.meta.data import Scope
from spflow.modules.outer_product import OuterProduct
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_data, make_leaf
from spflow.modules.leaf import Normal
import torch

cls_values = [OuterProduct, ElementwiseProduct]
in_channels_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]
is_single_input_values = [True, False]
params = list(product(in_channels_values, out_channels_values, out_features_values, is_single_input_values))


def make_module(cls, out_features: int, in_channels: int, split_method=None, split_indices=None, scopes=None):
    if split_method is not None:
        inputs = make_leaf(cls=Normal, out_channels=in_channels, out_features=out_features)

        if split_method == "split_indices":
            split_point = out_features // 2
            split_indices_a = list(range(split_point))
            split_indices_b = list(range(split_point, out_features))
            split_indices = (split_indices_a, split_indices_b)
    else:
        if scopes is None:
            scope_a = Scope(list(range(out_features)))
            scope_b = Scope(list(range(out_features, out_features * 2)))
        else:
            scope_a, scope_b = scopes
        inputs_a = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_a)
        inputs_b = make_leaf(cls=Normal, out_channels=in_channels, scope=scope_b)
        inputs = [inputs_a, inputs_b]

    return cls(inputs=inputs, split_method=split_method, split_indices=split_indices)


@pytest.mark.parametrize(
    "cls,in_channels,out_features,split_method",
    product(cls_values, in_channels_values, [2, 6], ["random", "split_indices"]),
)
def test_log_likelihood_single_input(cls, in_channels: int, out_features: int, split_method: str):
    module = make_module(
        cls=cls, out_features=out_features, in_channels=in_channels, split_method=split_method
    )

    data = make_data(cls=Normal, out_features=out_features)
    lls = log_likelihood(module, data)
    assert lls.shape == (data.shape[0], out_features // 2, module.out_channels)


@pytest.mark.parametrize(
    "cls,in_channels,out_features", product(cls_values, in_channels_values, out_features_values)
)
def test_log_likelihood_two_inputs(cls, in_channels: int, out_features: int):
    module = make_module(cls=cls, out_features=out_features, in_channels=in_channels, split_method=None)
    data = make_data(cls=Normal, out_features=out_features * 2)
    lls = log_likelihood(module, data)
    assert lls.shape == (data.shape[0], out_features, module.out_channels)


@pytest.mark.parametrize("cls,out_features", product(cls_values, out_features_values))
def test_log_likelihood_two_inputs_broadcasting_channels(cls, out_features: int):
    # Define the scopes
    in_channels_a = 1
    in_channels_b = 3
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))

    # Define the inputs
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels_a, scope=scope_a)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels_b, scope=scope_b)
    inputs = [inputs_a, inputs_b]

    # Create the module
    module = cls(inputs=inputs)

    # Create the data
    data = make_data(cls=Normal, out_features=out_features * 2)

    # Compute the log-likelihood
    lls = log_likelihood(module, data)
    assert lls.shape == (data.shape[0], out_features, module.out_channels)


@pytest.mark.parametrize(
    "cls,in_channels,out_features,split_method",
    product(cls_values, in_channels_values, [2, 6], ["random", "split_indices"]),
)
def test_sample_single_inputs(cls, in_channels: int, out_features: int, split_method: str):
    n_samples = 10
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    module = make_module(
        cls=cls, out_features=out_features, in_channels=in_channels, split_method=split_method
    )

    data = torch.full((n_samples, out_features), torch.nan)
    sampling_ctx.output_ids = torch.randint(
        low=0, high=module.out_channels, size=(n_samples, out_features // 2)
    )
    samples = sample(module, data, sampling_ctx=sampling_ctx)

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "cls,in_channels,out_features", product(cls_values, in_channels_values, out_features_values)
)
def test_sample_two_inputs(cls, in_channels: int, out_features: int):
    n_samples = 5
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    module = make_module(cls=cls, out_features=out_features, in_channels=in_channels, split_method=None)

    data = torch.full((n_samples, out_features * 2), torch.nan)
    sampling_ctx.output_ids = torch.randint(
        low=0, high=module.out_channels, size=(n_samples, module.out_features)
    )
    samples = sample(module, data, sampling_ctx=sampling_ctx)

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("cls,out_features", product(cls_values, out_features_values))
def test_sample_two_inputs_broadcasting_channels(cls, out_features: int):
    # Define the scopes
    in_channels_a = 1
    in_channels_b = 3
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))

    # Define the inputs
    inputs_a = make_leaf(cls=Normal, out_channels=in_channels_a, scope=scope_a)
    inputs_b = make_leaf(cls=Normal, out_channels=in_channels_b, scope=scope_b)
    inputs = [inputs_a, inputs_b]

    # Create the module
    module = cls(inputs=inputs)

    n_samples = 5
    data = torch.full((n_samples, out_features * 2), torch.nan)
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)
    sampling_ctx.output_ids = torch.randint(
        low=0, high=module.out_channels, size=(n_samples, module.out_features)
    )
    samples = sample(module, data, sampling_ctx=sampling_ctx)

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "cls,in_channels,out_features,split_method",
    product(cls_values, in_channels_values, [2, 6], ["random", "split_indices"]),
)
def test_scopes_single_input(cls, in_channels: int, out_features: int, split_method: str):
    module = make_module(
        cls=cls, out_features=out_features, in_channels=in_channels, split_method=split_method
    )
    assert module.scope.query == list(range(out_features))


@pytest.mark.parametrize(
    "cls,in_channels,out_features", product(cls_values, in_channels_values, out_features_values)
)
def test_scopes_two_inputs(cls, in_channels: int, out_features: int):
    module = make_module(cls=cls, out_features=out_features, in_channels=in_channels, split_method=None)
    assert module.scope.query == list(range(out_features * 2))


@pytest.mark.parametrize(
    "cls,in_channels,out_features,split_method",
    product(cls_values, in_channels_values, [2, 6], ["random", "split_indices", None]),
)
def test_expectation_maximization(cls, in_channels: int, out_features: int, split_method: str):
    module = make_module(
        cls=cls, out_features=out_features, in_channels=in_channels, split_method=split_method
    )
    data = make_data(cls=Normal, out_features=out_features * 2 if split_method is None else out_features)
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
            split_method=None,
            scopes=(Scope(range(out_features)), Scope(range(out_features))),
        )


@pytest.mark.parametrize("cls", cls_values)
def test_invalid_split_method_should_be_none(cls):
    with pytest.raises(InvalidParameterCombinationError):
        cls(
            inputs=[
                make_normal_leaf(scope=Scope([0]), out_channels=3),
                make_normal_leaf(scope=Scope([1]), out_channels=3),
            ],
            split_method="split_indices",
        )


@pytest.mark.parametrize("cls", cls_values)
def test_invalid_split_method_should_not_be_none(cls):
    with pytest.raises(InvalidParameterCombinationError):
        cls(
            inputs=make_normal_leaf(scope=Scope([0, 1]), out_channels=3),
            split_method=None,
        )


@pytest.mark.parametrize("cls", cls_values)
def test_invalid_split_indices_should_not_be_none(cls):
    with pytest.raises(InvalidParameterCombinationError):
        cls(
            inputs=make_normal_leaf(scope=Scope([0, 1]), out_channels=3),
            split_method="split_indices",
            split_indices=None,
        )


@pytest.mark.parametrize("cls", cls_values)
def test_invalid_split_indices_should_be_none(cls):
    with pytest.raises(InvalidParameterCombinationError):
        cls(
            inputs=make_normal_leaf(scope=Scope([0, 1]), out_channels=3),
            split_method="random",
            split_indices=([0, 1], [2, 3]),
        )


@pytest.mark.parametrize("cls", cls_values)
def test_invalid_out_features_number(cls):
    with pytest.raises(ValueError):
        cls(
            inputs=[
                make_normal_leaf(scope=Scope([0]), out_channels=3),
                make_normal_leaf(scope=Scope([1, 2, 3]), out_channels=3),
            ],
        )


@pytest.mark.parametrize("cls", cls_values)
def test_invalid_split_method_should_be_none(cls):
    with pytest.raises(InvalidParameterCombinationError):
        cls(
            inputs=[
                make_normal_leaf(scope=Scope([0]), out_channels=3),
                make_normal_leaf(scope=Scope([1]), out_channels=3),
            ],
            split_method="split_indices",
        )


@pytest.mark.parametrize("cls", cls_values)
def test_invalid_split_indices_too_few(cls):
    with pytest.raises(ValueError):
        cls(
            inputs=make_normal_leaf(scope=Scope([0, 1, 2, 3]), out_channels=3),
            split_method="split_indices",
            split_indices=([0], [2]),
        )


@pytest.mark.parametrize("cls", cls_values)
def test_uneven_input_features(cls):
    with pytest.raises(ValueError):
        cls(
            inputs=make_normal_leaf(scope=Scope([0, 1, 2]), out_channels=3),
            split_method="split_indices",
            split_indices=([0], [2]),
        )


@pytest.mark.parametrize("cls,prune", product(cls_values, [True, False]))
def test_marginalize_single_input(cls, prune: bool):
    # TODO: implement marginalization
    pass
