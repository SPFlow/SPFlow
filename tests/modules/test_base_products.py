from itertools import product

import pytest
import torch

from spflow.exceptions import ScopeError, ShapeError
from spflow.learn import expectation_maximization
from spflow.meta import Scope
from spflow.modules.products import ElementwiseProduct
from spflow.modules.products.outer_product import OuterProduct
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import make_normal_leaf, make_data, make_leaf, DummyLeaf

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
    inputs_a = make_leaf(
        cls=DummyLeaf, out_channels=in_channels, scope=scope_a, num_repetitions=num_repetitions
    )
    inputs_b = make_leaf(
        cls=DummyLeaf, out_channels=in_channels, scope=scope_b, num_repetitions=num_repetitions
    )
    inputs_c = make_leaf(
        cls=DummyLeaf, out_channels=in_channels, scope=scope_c, num_repetitions=num_repetitions
    )
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
    data = make_data(cls=DummyLeaf, out_features=out_features * len(module.inputs))
    lls = module.log_likelihood(data)
    # Always expect 4D output [batch, features, channels, num_reps]
    assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)


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
    inputs_a = make_leaf(cls=DummyLeaf, out_channels=in_channels_a, scope=scope_a, num_repetitions=num_reps)
    inputs_b = make_leaf(cls=DummyLeaf, out_channels=in_channels_b, scope=scope_b, num_repetitions=num_reps)
    inputs_c = make_leaf(cls=DummyLeaf, out_channels=in_channels_c, scope=scope_c, num_repetitions=num_reps)
    inputs = [inputs_a, inputs_b, inputs_c]

    # Create the module
    module = cls(inputs=inputs)

    # Create the data
    data = make_data(cls=DummyLeaf, out_features=out_features * len(module.inputs))

    # Compute the log-likelihood
    lls = module.log_likelihood(data)
    # Always expect 4D output [batch, features, channels, num_reps]
    assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)


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
    mask = torch.full((n_samples, module.out_shape.features), True, dtype=torch.bool)
    channel_index = torch.randint(
        low=0, high=module.out_shape.channels, size=(n_samples, module.out_shape.features)
    )
    # Always set repetition_index since num_reps is never None
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("cls", cls_values)
def test_sample_differentiable_smoke(cls):
    n_samples = 64
    out_features = 4
    num_reps = 1
    module = make_module(
        cls=cls,
        out_features=out_features,
        in_channels=3,
        num_repetitions=num_reps,
    )

    data = torch.full((n_samples, out_features * len(module.inputs)), torch.nan)
    channel_index_int = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(n_samples, module.out_shape.features),
    )
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(channel_index_int, dim=-1, dim_size=module.out_shape.channels),
        mask=torch.ones((n_samples, module.out_shape.features), dtype=torch.bool),
        repetition_index=to_one_hot(torch.zeros((n_samples,), dtype=torch.long), dim=-1, dim_size=num_reps),
        is_differentiable=True,
        hard=True,
    )
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == data.shape
    assert torch.isfinite(samples[:, module.scope.query]).all()


@pytest.mark.parametrize("cls", cls_values)
def test_sample_differentiable_equals_non_diff_sampling(cls):
    n_samples = 48
    out_features = 4
    num_reps = 1
    module = make_module(
        cls=cls,
        out_features=out_features,
        in_channels=3,
        num_repetitions=num_reps,
    )
    channel_index = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(n_samples, module.out_shape.features),
    )
    mask = torch.ones((n_samples, module.out_shape.features), dtype=torch.bool)
    repetition_index = torch.zeros((n_samples,), dtype=torch.long)
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index.clone(),
        mask=mask.clone(),
        repetition_index=repetition_index.clone(),
    )
    sampling_ctx_b = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=module.out_shape.channels),
        mask=mask.clone(),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_differentiable=True,
        hard=True,
    )

    torch.manual_seed(1337)
    samples_a = module._sample(
        data=torch.full((n_samples, out_features * len(module.inputs)), torch.nan),
        sampling_ctx=sampling_ctx_a,
        cache=Cache(),
    )
    torch.manual_seed(1337)
    samples_b = module._sample(
        data=torch.full((n_samples, out_features * len(module.inputs)), torch.nan),
        sampling_ctx=sampling_ctx_b,
        cache=Cache(),
    )

    torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)


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
    inputs_a = make_leaf(cls=DummyLeaf, out_channels=in_channels_a, scope=scope_a, num_repetitions=num_reps)
    inputs_b = make_leaf(cls=DummyLeaf, out_channels=in_channels_b, scope=scope_b, num_repetitions=num_reps)
    inputs_c = make_leaf(cls=DummyLeaf, out_channels=in_channels_c, scope=scope_c, num_repetitions=num_reps)
    inputs = [inputs_a, inputs_b, inputs_c]

    # Create the module
    module = cls(inputs=inputs)

    n_samples = 5
    data = torch.full((n_samples, out_features * len(module.inputs)), torch.nan)
    channel_index = torch.zeros((n_samples, module.out_shape.features), dtype=torch.long)
    mask = torch.full((n_samples, module.out_shape.features), True, dtype=torch.bool)
    # Always set repetition_index since num_reps is never None
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize("cls", cls_values)
def test_sample_differentiable_broadcasting_channels(cls):
    out_features = 3
    n_samples = 32
    num_reps = 1
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))
    scope_c = Scope(list(range(out_features * 2, out_features * 3)))
    inputs_a = make_leaf(cls=DummyLeaf, out_channels=1, scope=scope_a, num_repetitions=num_reps)
    inputs_b = make_leaf(cls=DummyLeaf, out_channels=3, scope=scope_b, num_repetitions=num_reps)
    inputs_c = make_leaf(cls=DummyLeaf, out_channels=3, scope=scope_c, num_repetitions=num_reps)
    module = cls(inputs=[inputs_a, inputs_b, inputs_c])

    data = torch.full((n_samples, out_features * len(module.inputs)), torch.nan)
    channel_index_int = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(n_samples, module.out_shape.features),
    )
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(channel_index_int, dim=-1, dim_size=module.out_shape.channels),
        mask=torch.ones((n_samples, module.out_shape.features), dtype=torch.bool),
        repetition_index=to_one_hot(torch.zeros((n_samples,), dtype=torch.long), dim=-1, dim_size=num_reps),
        is_differentiable=True,
        hard=True,
    )
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == data.shape
    assert torch.isfinite(samples[:, module.scope.query]).all()


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
    data = make_data(cls=DummyLeaf, out_features=out_features * len(module.inputs))
    locs_before = [inp.loc.detach().clone() for inp in module.inputs]
    scales_before = [inp.scale.detach().clone() for inp in module.inputs]

    max_steps = 2
    ll_history = expectation_maximization(module, data, max_steps=max_steps)
    assert ll_history.ndim == 1
    assert 1 <= ll_history.numel() <= max_steps
    assert ll_history.isfinite().all()

    for i, inp in enumerate(module.inputs):
        assert not torch.equal(inp.loc, locs_before[i])
        assert not torch.equal(inp.scale, scales_before[i])
        torch.testing.assert_close(inp.loc, torch.zeros_like(inp.loc))
        torch.testing.assert_close(inp.scale, torch.ones_like(inp.scale))


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


def test_sample_does_not_mutate_parent_sampling_context():
    module = make_module(
        cls=ElementwiseProduct,
        out_features=2,
        in_channels=3,
        num_repetitions=1,
    )
    n_samples = 5
    data = torch.full((n_samples, 2 * len(module.inputs)), torch.nan)
    channel_index = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(n_samples, module.out_shape.features),
    )
    mask = torch.tensor(
        [[True, True], [True, False], [False, True], [True, True], [True, True]],
        dtype=torch.bool,
    )
    sampling_ctx = SamplingContext(
        channel_index=channel_index.clone(),
        mask=mask.clone(),
        repetition_index=torch.zeros((n_samples,), dtype=torch.long),
    )
    channel_before = sampling_ctx.channel_index.clone()
    mask_before = sampling_ctx.mask.clone()

    module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert torch.equal(sampling_ctx.channel_index, channel_before)
    assert torch.equal(sampling_ctx.mask, mask_before)


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
        with pytest.raises(ShapeError):
            product(inputs=leaf_layer)
    else:
        prod = product(inputs=leaf_layer)
        assert prod is not None


def test_elementwise_product_feature_to_scope():
    """Test feature_to_scope property for ElementwiseProduct module.

    ElementwiseProduct performs element-wise joining of corresponding features from inputs.
    Output shape should be (out_features, num_repetitions).
    """

    # Test with single repetition
    out_features = 3
    out_channels = 2
    num_reps = 1

    # Create inputs with disjoint scopes
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))

    leaf_a = make_normal_leaf(scope=scope_a, out_channels=out_channels, num_repetitions=num_reps)
    leaf_b = make_normal_leaf(scope=scope_b, out_channels=out_channels, num_repetitions=num_reps)

    elem_prod = ElementwiseProduct(inputs=[leaf_a, leaf_b])

    # Get feature_to_scope
    feature_scopes = elem_prod.feature_to_scope

    # Validate shape: should be (out_features, num_repetitions)
    assert feature_scopes.shape == (
        out_features,
        num_reps,
    ), f"Expected shape ({out_features}, {num_reps}), got {feature_scopes.shape}"

    # Validate all elements are Scope objects
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Validate scope content: each output feature should be the join of corresponding input features
    for i in range(out_features):
        for r in range(num_reps):
            # Element-wise product joins corresponding features
            expected_scope = Scope.join_all([leaf_a.feature_to_scope[i, r], leaf_b.feature_to_scope[i, r]])
            assert (
                feature_scopes[i, r] == expected_scope
            ), f"Feature {i}, Rep {r}: expected {expected_scope}, got {feature_scopes[i, r]}"
            # Each output feature should contain exactly 2 input features (one from each input)
            assert (
                len(feature_scopes[i, r].query) == 2
            ), f"Feature {i}, Rep {r}: expected 2 features in scope, got {len(feature_scopes[i, r].query)}"


def test_elementwise_product_feature_to_scope_multiple_repetitions():
    """Test feature_to_scope with multiple repetitions for ElementwiseProduct module."""

    # Test with multiple repetitions
    out_features = 4
    out_channels = 3
    num_reps = 3

    # Create inputs with disjoint scopes
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))
    scope_c = Scope(list(range(out_features * 2, out_features * 3)))

    leaf_a = make_normal_leaf(scope=scope_a, out_channels=out_channels, num_repetitions=num_reps)
    leaf_b = make_normal_leaf(scope=scope_b, out_channels=out_channels, num_repetitions=num_reps)
    leaf_c = make_normal_leaf(scope=scope_c, out_channels=out_channels, num_repetitions=num_reps)

    elem_prod = ElementwiseProduct(inputs=[leaf_a, leaf_b, leaf_c])

    # Get feature_to_scope
    feature_scopes = elem_prod.feature_to_scope

    # Validate shape: should be (out_features, num_repetitions)
    assert feature_scopes.shape == (
        out_features,
        num_reps,
    ), f"Expected shape ({out_features}, {num_reps}), got {feature_scopes.shape}"

    # Validate all elements are Scope objects
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Validate scope content for each repetition
    for i in range(out_features):
        for r in range(num_reps):
            # Element-wise product joins corresponding features from all 3 inputs
            expected_scope = Scope.join_all(
                [leaf_a.feature_to_scope[i, r], leaf_b.feature_to_scope[i, r], leaf_c.feature_to_scope[i, r]]
            )
            assert (
                feature_scopes[i, r] == expected_scope
            ), f"Feature {i}, Rep {r}: expected {expected_scope}, got {feature_scopes[i, r]}"
            # Each output feature should contain 3 input features (one from each input)
            assert (
                len(feature_scopes[i, r].query) == 3
            ), f"Feature {i}, Rep {r}: expected 3 features in scope, got {len(feature_scopes[i, r].query)}"


def test_outer_product_feature_to_scope():
    """Test feature_to_scope property for OuterProduct module.

    OuterProduct performs Cartesian product expansion: all combinations of input features.
    With 2 inputs each having 2 features, output has 2x2=4 features.
    """
    from itertools import product as iter_product

    # Test with single repetition and 2 inputs
    out_features = 2
    out_channels_a = 2
    out_channels_b = 3
    num_reps = 1

    # Create inputs with disjoint scopes
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))

    leaf_a = make_normal_leaf(scope=scope_a, out_channels=out_channels_a, num_repetitions=num_reps)
    leaf_b = make_normal_leaf(scope=scope_b, out_channels=out_channels_b, num_repetitions=num_reps)

    outer_prod = OuterProduct(inputs=[leaf_a, leaf_b])

    # Get feature_to_scope
    feature_scopes = outer_prod.feature_to_scope

    # OuterProduct creates Cartesian product: 2 features x 2 features = 4 output features
    expected_num_features = out_features * out_features

    # Validate shape: should be (expected_num_features, num_repetitions)
    assert feature_scopes.shape == (
        expected_num_features,
        num_reps,
    ), f"Expected shape ({expected_num_features}, {num_reps}), got {feature_scopes.shape}"

    # Validate all elements are Scope objects
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Validate Cartesian product logic
    # The outer product should create all combinations of features from both inputs
    for r in range(num_reps):
        expected_combinations = list(
            iter_product(leaf_a.feature_to_scope[:, r], leaf_b.feature_to_scope[:, r])
        )

        for i, (scope_a_elem, scope_b_elem) in enumerate(expected_combinations):
            expected_scope = Scope.join_all([scope_a_elem, scope_b_elem])
            assert (
                feature_scopes[i, r] == expected_scope
            ), f"Feature {i}, Rep {r}: expected {expected_scope}, got {feature_scopes[i, r]}"
            # Each output feature should contain 2 input features (one from each input)
            assert (
                len(feature_scopes[i, r].query) == 2
            ), f"Feature {i}, Rep {r}: expected 2 features in scope, got {len(feature_scopes[i, r].query)}"


def test_outer_product_feature_to_scope_multiple_repetitions():
    """Test feature_to_scope with multiple repetitions for OuterProduct module.

    With 3 inputs each having 2 features, output has 2x2x2=8 features (Cartesian product).
    """
    from itertools import product as iter_product

    # Test with multiple repetitions and 3 inputs
    out_features = 2  # Each input has 2 features
    out_channels = 2
    num_reps = 2

    # Create inputs with disjoint scopes
    scope_a = Scope(list(range(out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))
    scope_c = Scope(list(range(out_features * 2, out_features * 3)))

    leaf_a = make_normal_leaf(scope=scope_a, out_channels=out_channels, num_repetitions=num_reps)
    leaf_b = make_normal_leaf(scope=scope_b, out_channels=out_channels, num_repetitions=num_reps)
    leaf_c = make_normal_leaf(scope=scope_c, out_channels=out_channels, num_repetitions=num_reps)

    outer_prod = OuterProduct(inputs=[leaf_a, leaf_b, leaf_c])

    # Get feature_to_scope
    feature_scopes = outer_prod.feature_to_scope

    # Cartesian product: 2 x 2 x 2 = 8 output features
    expected_num_features = out_features * out_features * out_features

    # Validate shape: should be (expected_num_features, num_repetitions)
    assert feature_scopes.shape == (
        expected_num_features,
        num_reps,
    ), f"Expected shape ({expected_num_features}, {num_reps}), got {feature_scopes.shape}"

    # Validate all elements are Scope objects
    assert all(isinstance(s, Scope) for s in feature_scopes.flatten()), "All elements should be Scope objects"

    # Validate scope content for each repetition
    for r in range(num_reps):
        # Create all Cartesian product combinations
        expected_combinations = list(
            iter_product(
                leaf_a.feature_to_scope[:, r], leaf_b.feature_to_scope[:, r], leaf_c.feature_to_scope[:, r]
            )
        )

        for i, (scope_a_elem, scope_b_elem, scope_c_elem) in enumerate(expected_combinations):
            expected_scope = Scope.join_all([scope_a_elem, scope_b_elem, scope_c_elem])
            assert (
                feature_scopes[i, r] == expected_scope
            ), f"Feature {i}, Rep {r}: expected {expected_scope}, got {feature_scopes[i, r]}"
            # Each output feature should contain 3 input features (one from each input)
            assert (
                len(feature_scopes[i, r].query) == 3
            ), f"Feature {i}, Rep {r}: expected 3 features in scope, got {len(feature_scopes[i, r].query)}"
