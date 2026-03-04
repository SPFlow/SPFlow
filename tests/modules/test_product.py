import pytest
import torch

from spflow.modules.products import Product
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data


def make_product(in_channels=None, out_features=None, inputs=None, num_repetitions=None):
    if inputs is None:
        inputs = make_normal_leaf(
            out_features=out_features, out_channels=in_channels, num_repetitions=num_repetitions
        )
    return Product(inputs=inputs)


# Cross-module product contracts moved to:
# - test_product_contract_loglikelihood.py
# - test_product_contract_sampling.py
# - test_product_contract_training_marginalization.py


def test_constructor():
    in_channels = 3
    out_features = 4
    num_reps = 5

    module = make_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)

    assert isinstance(module, Product)
    assert tuple(module.scope.query) == tuple(range(out_features))
    assert module.in_shape.features == out_features
    assert module.out_shape.features == 1
    assert module.out_shape.channels == in_channels
    assert module.out_shape.repetitions == num_reps


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
        num_repetitions=num_reps,
        mean=mean,
        std=std,
    )
    normal_layer_b1 = make_normal_leaf(
        scope=[0, 1],
        out_features=out_features / 2,
        out_channels=in_channels,
        num_repetitions=num_reps,
        mean=mean[0:2, :, :],
        std=std[0:2, :, :],
    )
    normal_layer_b2 = make_normal_leaf(
        scope=[2, 3],
        out_features=out_features / 2,
        out_channels=in_channels,
        num_repetitions=num_reps,
        mean=mean[2:4, :, :],
        std=std[2:4, :, :],
    )

    module_a = Product(inputs=normal_layer_a)

    module_b = Product(inputs=[normal_layer_b1, normal_layer_b2])

    # Guard against regressions where list-input composition changes semantics.

    data = make_normal_data(out_features=out_features)

    ll_a = module_a.log_likelihood(data)
    ll_b = module_b.log_likelihood(data)

    torch.testing.assert_close(ll_a, ll_b, rtol=1e-5, atol=1e-6)

    # Sampling should stay equivalent for both construction paths under shared routing.

    n_samples = 10

    data_a = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(low=0, high=out_channels, size=(n_samples, module_a.out_shape.features))
    mask = torch.full((n_samples, module_a.out_shape.features), True)
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index, mask=mask, repetition_index=repetition_index, is_mpe=True
    )

    data_b = torch.full((n_samples, out_features), torch.nan)

    sampling_ctx_b = SamplingContext(
        channel_index=channel_index, mask=mask, repetition_index=repetition_index, is_mpe=True
    )

    samples_a = module_a._sample(data=data_a, sampling_ctx=sampling_ctx_a, cache=Cache())
    samples_b = module_b._sample(data=data_b, sampling_ctx=sampling_ctx_b, cache=Cache())

    torch.testing.assert_close(samples_a, samples_b, rtol=0.0, atol=0.0)


def test_feature_to_scope():
    """Test feature_to_scope property for Product module.

    Product joins all input scopes into a single combined scope per repetition.
    Output shape should be (1, num_repetitions).
    """
    from spflow.meta import Scope

    out_features = 6
    out_channels = 3
    num_reps = 1

    scope = Scope(list(range(out_features)))
    leaf = make_normal_leaf(scope=scope, out_channels=out_channels, num_repetitions=num_reps)
    product = Product(inputs=leaf)

    feature_scopes = product.feature_to_scope

    assert feature_scopes.shape == (
        1,
        num_reps,
    ), f"Expected shape (1, {num_reps}), got {feature_scopes.shape}"

    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    expected_scope = Scope.join_all(leaf.feature_to_scope[:, 0])
    assert feature_scopes[0, 0] == expected_scope, f"Expected {expected_scope}, got {feature_scopes[0, 0]}"

    # Product must not drop feature IDs when collapsing to one output feature.
    assert set(feature_scopes[0, 0].query) == set(
        range(out_features)
    ), "Joined scope should contain all input features"


def test_feature_to_scope_multiple_repetitions():
    """Test feature_to_scope with multiple repetitions for Product module."""
    from spflow.meta import Scope

    out_features = 4
    out_channels = 2
    num_reps = 3

    scope = Scope(list(range(out_features)))
    leaf = make_normal_leaf(scope=scope, out_channels=out_channels, num_repetitions=num_reps)
    product = Product(inputs=leaf)

    feature_scopes = product.feature_to_scope

    assert feature_scopes.shape == (
        1,
        num_reps,
    ), f"Expected shape (1, {num_reps}), got {feature_scopes.shape}"

    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Repetition-specific bookkeeping must not alter the represented variable set.
    for r in range(num_reps):
        expected_scope = Scope.join_all(leaf.feature_to_scope[:, r])
        assert (
            feature_scopes[0, r] == expected_scope
        ), f"Repetition {r}: expected {expected_scope}, got {feature_scopes[0, r]}"
        assert set(feature_scopes[0, r].query) == set(
            range(out_features)
        ), f"Repetition {r}: joined scope should contain all features"


def test_feature_to_scope_multiple_inputs():
    """Test feature_to_scope with multiple input modules for Product module."""
    from spflow.meta import Scope

    out_features_1 = 3
    out_features_2 = 2
    out_channels = 2
    num_reps = 2

    scope_1 = Scope(list(range(out_features_1)))
    scope_2 = Scope(list(range(out_features_1, out_features_1 + out_features_2)))

    leaf_1 = make_normal_leaf(scope=scope_1, out_channels=out_channels, num_repetitions=num_reps)
    leaf_2 = make_normal_leaf(scope=scope_2, out_channels=out_channels, num_repetitions=num_reps)

    product = Product(inputs=[leaf_1, leaf_2])

    feature_scopes = product.feature_to_scope

    assert feature_scopes.shape == (
        1,
        num_reps,
    ), f"Expected shape (1, {num_reps}), got {feature_scopes.shape}"

    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Disjoint child scopes should union cleanly in the parent scope map.
    total_features = out_features_1 + out_features_2
    for r in range(num_reps):
        assert set(feature_scopes[0, r].query) == set(
            range(total_features)
        ), f"Repetition {r}: joined scope should contain all {total_features} features"
