from itertools import product

import pytest
import torch

from spflow.learn import expectation_maximization
from spflow.modules.products import Product
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data

# Constants
in_channels_values = [1, 3]
out_features_values = [1, 4]
num_repetitions = [1, 5]
params = list(product(in_channels_values, out_features_values, num_repetitions))


def make_product(in_channels=None, out_features=None, inputs=None, num_repetitions=None):
    if inputs is None:
        inputs = make_normal_leaf(
            out_features=out_features, out_channels=in_channels, num_repetitions=num_repetitions
        )
    return Product(inputs=inputs)


@pytest.mark.parametrize("in_channels,out_features,num_reps", params)
def test_log_likelihood(in_channels: int, out_features: int, num_reps, device):
    product_layer = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    data = make_normal_data(out_features=out_features)
    lls = product_layer.log_likelihood(data)
    # Always expect 4D output
    assert lls.shape == (data.shape[0], 1, product_layer.out_channels, num_reps)


@pytest.mark.parametrize("in_channels,out_features,num_reps", params)
def test_sample(in_channels: int, out_features: int, num_reps, device):
    n_samples = 10
    product_layer = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    for i in range(product_layer.out_channels):
        data = torch.full((n_samples, out_features), torch.nan)
        channel_index = torch.full((n_samples, out_features), fill_value=i)
        mask = torch.full((n_samples, out_features), True, dtype=torch.bool)
        if num_reps is not None:
            repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
        else:
            repetition_index = None
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )
        samples = product_layer.sample(data=data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, product_layer.scope.query]
        assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("in_channels,out_features,num_reps", params)
def test_expectation_maximization(in_channels: int, out_features: int, num_reps, device):
    product_layer = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    data = make_normal_data(out_features=out_features)
    with torch.autograd.set_detect_anomaly(True):
        expectation_maximization(product_layer, data, max_steps=10)


def test_constructor():
    pass


@pytest.mark.parametrize(
    "prune,in_channels,marg_rvs,num_reps",
    product(
        [True, False],
        in_channels_values,
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        num_repetitions,
    ),
)
def test_marginalize(prune, in_channels: int, marg_rvs: list[int], num_reps, device):
    out_features = 3
    module = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)

    # Marginalize scope
    marginalized_module = module.marginalize(marg_rvs, prune=prune)

    if len(marg_rvs) == out_features:
        assert marginalized_module is None
        return

    if prune and len(marg_rvs) == (out_features - 1):
        # If pruning is active and only one scope is left, the (pruned) input module should be returned
        assert isinstance(marginalized_module, type(module.inputs))

    # Scope query should not contain marginalized rv
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0


def test_multiple_inputs():
    in_channels = 2
    out_channels = 2
    out_features = 4
    num_reps = 5

    mean = torch.rand((out_features, out_channels, num_reps))
    std = torch.rand((out_features, out_channels, num_reps))

    normal_layer_a = make_normal_leaf(
        scope=[0, 1, 2, 3],
        out_features=out_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
        mean=mean,
        std=std,
    )
    normal_layer_b1 = make_normal_leaf(
        scope=[0, 1],
        out_features=out_features / 2,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
        mean=mean[0:2, :, :],
        std=std[0:2, :, :],
    )
    normal_layer_b2 = make_normal_leaf(
        scope=[2, 3],
        out_features=out_features / 2,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
        mean=mean[2:4, :, :],
        std=std[2:4, :, :],
    )

    module_a = Product(inputs=normal_layer_a)

    module_b = Product(inputs=[normal_layer_b1, normal_layer_b2])

    # test log likelihood

    data = make_normal_data(out_features=out_features)

    ll_a = module_a.log_likelihood(data)
    ll_b = module_b.log_likelihood(data)

    assert torch.allclose(ll_a, ll_b)

    # test sampling

    n_samples = 10

    data_a = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(low=0, high=out_channels, size=(n_samples, out_features))
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


def test_feature_to_scope():
    """Test feature_to_scope property for Product module.

    Product joins all input scopes into a single combined scope per repetition.
    Output shape should be (1, num_repetitions).
    """
    from spflow.meta import Scope

    # Test with single repetition
    out_features = 6
    out_channels = 3
    num_reps = 1

    # Create input with known scope
    scope = Scope(list(range(out_features)))
    leaf = make_normal_leaf(scope=scope, out_channels=out_channels, num_repetitions=num_reps)
    product = Product(inputs=leaf)

    # Get feature_to_scope
    feature_scopes = product.feature_to_scope

    # Validate shape: should be (1, num_repetitions) since Product outputs 1 feature
    assert feature_scopes.shape == (1, num_reps), f"Expected shape (1, {num_reps}), got {feature_scopes.shape}"

    # Validate all elements are Scope objects
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Validate scope content: should be the joined scope of all input features
    expected_scope = Scope.join_all(leaf.feature_to_scope[:, 0])
    assert feature_scopes[0, 0] == expected_scope, f"Expected {expected_scope}, got {feature_scopes[0, 0]}"

    # Verify the joined scope contains all input features
    assert set(feature_scopes[0, 0].query) == set(range(out_features)), "Joined scope should contain all input features"


def test_feature_to_scope_multiple_repetitions():
    """Test feature_to_scope with multiple repetitions for Product module."""
    from spflow.meta import Scope

    # Test with multiple repetitions
    out_features = 4
    out_channels = 2
    num_reps = 3

    scope = Scope(list(range(out_features)))
    leaf = make_normal_leaf(scope=scope, out_channels=out_channels, num_repetitions=num_reps)
    product = Product(inputs=leaf)

    # Get feature_to_scope
    feature_scopes = product.feature_to_scope

    # Validate shape: should be (1, num_repetitions)
    assert feature_scopes.shape == (1, num_reps), f"Expected shape (1, {num_reps}), got {feature_scopes.shape}"

    # Validate all elements are Scope objects
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Validate each repetition has the same joined scope
    for r in range(num_reps):
        expected_scope = Scope.join_all(leaf.feature_to_scope[:, r])
        assert feature_scopes[0, r] == expected_scope, f"Repetition {r}: expected {expected_scope}, got {feature_scopes[0, r]}"
        assert set(feature_scopes[0, r].query) == set(range(out_features)), f"Repetition {r}: joined scope should contain all features"


def test_feature_to_scope_multiple_inputs():
    """Test feature_to_scope with multiple input modules for Product module."""
    from spflow.meta import Scope

    # Create two inputs with disjoint scopes
    out_features_1 = 3
    out_features_2 = 2
    out_channels = 2
    num_reps = 2

    scope_1 = Scope(list(range(out_features_1)))
    scope_2 = Scope(list(range(out_features_1, out_features_1 + out_features_2)))

    leaf_1 = make_normal_leaf(scope=scope_1, out_channels=out_channels, num_repetitions=num_reps)
    leaf_2 = make_normal_leaf(scope=scope_2, out_channels=out_channels, num_repetitions=num_reps)

    # Product with multiple inputs (they will be concatenated)
    product = Product(inputs=[leaf_1, leaf_2])

    # Get feature_to_scope
    feature_scopes = product.feature_to_scope

    # Validate shape: should be (1, num_repetitions)
    assert feature_scopes.shape == (1, num_reps), f"Expected shape (1, {num_reps}), got {feature_scopes.shape}"

    # Validate all elements are Scope objects
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Validate scope content: should contain all features from both inputs
    total_features = out_features_1 + out_features_2
    for r in range(num_reps):
        assert set(feature_scopes[0, r].query) == set(range(total_features)), \
            f"Repetition {r}: joined scope should contain all {total_features} features"
