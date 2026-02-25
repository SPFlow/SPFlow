from itertools import product

import numpy as np
import pytest
import torch

from spflow.exceptions import (
    InvalidParameterCombinationError,
    InvalidWeightsError,
    MissingCacheError,
    ScopeError,
    ShapeError,
)
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.products import ElementwiseProduct
from spflow.modules.sums.elementwise_sum import ElementwiseSum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import (
    DummyLeaf,
    make_leaf,
    make_normal_data,
    make_normal_leaf,
)
from tests.utils.sampling_context_helpers import patch_simple_as_categorical_one_hot

in_channels_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]
num_repetitions = [1, 7]
invalid_constructor_cases = [(1, 1, 1, 1), (4, 5, 6, 7)]


def _randn(*size: int) -> torch.Tensor:
    return torch.randn(*size)


def _rand(*size: int) -> torch.Tensor:
    return torch.rand(*size)


def make_sum(
    in_channels=None, out_channels=None, out_features=None, weights=None, scopes=None, num_repetitions=None
):
    if isinstance(weights, list):
        weights = torch.tensor(weights)
        if weights.dim() == 1:
            weights = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        elif weights.dim() == 2:
            weights = weights.unsqueeze(2).unsqueeze(3)

    # Track num_repetitions for leaves separately from what we pass to ElementwiseSum
    leaf_num_reps = num_repetitions
    sum_num_reps = num_repetitions

    if weights is not None:
        out_features = weights.shape[0]
        # Derive num_repetitions from weights for creating leaves (if valid 5D weights)
        if weights.dim() == 5:
            leaf_num_reps = weights.shape[4]
        else:
            # Invalid weights shape - use num_repetitions if provided, else default to 1
            leaf_num_reps = num_repetitions if num_repetitions is not None else 1
        # ElementwiseSum doesn't accept num_repetitions when weights is provided
        sum_num_reps = None

    if scopes is None:
        scope_a = Scope(list(range(out_features)))
        scope_b = Scope(list(range(out_features)))
    else:
        scope_a, scope_b = scopes
    inputs_a = make_leaf(
        cls=DummyLeaf, out_channels=in_channels, scope=scope_a, num_repetitions=leaf_num_reps
    )
    inputs_b = make_leaf(
        cls=DummyLeaf, out_channels=in_channels, scope=scope_b, num_repetitions=leaf_num_reps
    )
    inputs = [inputs_a, inputs_b]

    return ElementwiseSum(
        out_channels=out_channels, inputs=inputs, weights=weights, num_repetitions=sum_num_reps
    )


# Cross-module behavioral contracts for the sum family moved to:
# - test_sum_contract_loglikelihood.py
# - test_sum_contract_sampling.py
# - test_sum_contract_training.py
# - test_sum_contract_marginalization.py
# - test_sum_contract_weights_and_constructor.py


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", invalid_constructor_cases)
def test_invalid_specification_of_out_channels_and_weights(
    in_channels: int, out_channels: int, out_features: int, num_reps
):
    weights = _rand((out_features, in_channels, out_channels, 2, num_reps))
    weights = weights / weights.sum(dim=1, keepdim=True)
    with pytest.raises(ShapeError):
        # Mismatched out_channels between weights and inputs
        out_channels_leaves = out_channels + 1
        ElementwiseSum(
            weights=weights,
            inputs=[
                make_normal_leaf(
                    out_features=out_features, out_channels=out_channels_leaves, num_repetitions=num_reps
                ),
                make_normal_leaf(
                    out_features=out_features, out_channels=out_channels_leaves, num_repetitions=num_reps
                ),
            ],
        )

    channels_a = max(2, in_channels)
    channels_b = channels_a + 1
    input_a = make_normal_leaf(out_features=out_features, out_channels=channels_a, num_repetitions=num_reps)
    input_b = make_normal_leaf(out_features=out_features, out_channels=channels_b, num_repetitions=num_reps)
    with pytest.raises(ShapeError):
        ElementwiseSum(out_channels=out_channels, inputs=[input_a, input_b])


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", invalid_constructor_cases)
def test_invalid_input_features_mismatch(in_channels: int, out_channels: int, out_features: int, num_reps):
    input_a = make_normal_leaf(
        out_features=out_features + 1, out_channels=in_channels, num_repetitions=num_reps
    )
    input_b = make_normal_leaf(
        out_features=out_features + 2, out_channels=in_channels, num_repetitions=num_reps
    )
    with pytest.raises(ShapeError):
        ElementwiseSum(out_channels=out_channels, inputs=[input_a, input_b])


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", invalid_constructor_cases)
def test_invalid_parameter_combination(in_channels: int, out_channels: int, out_features: int, num_reps):
    weights = _rand((out_features, in_channels, out_channels, num_reps)) + 1.0
    with pytest.raises(InvalidParameterCombinationError):
        make_sum(
            weights=weights, out_channels=out_channels, in_channels=in_channels, num_repetitions=num_reps
        )


@pytest.mark.parametrize(
    "in_channels,out_channels,out_features",
    product(in_channels_values, out_channels_values, out_features_values),
)
def test_same_scope_error(in_channels: int, out_channels: int, out_features: int):
    input_a = make_normal_leaf(scope=Scope(list(list(range(0, out_features)))), out_channels=in_channels)
    input_b = make_normal_leaf(
        scope=Scope(list(list(range(out_features, out_features * 2)))), out_channels=in_channels
    )
    with pytest.raises(ScopeError):
        ElementwiseSum(out_channels=out_channels, inputs=[input_a, input_b])


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


def test_feature_to_scope_basic():
    """Test that feature_to_scope correctly delegates to first input module."""
    out_features = 6
    in_channels = 3
    out_channels = 4
    num_reps = 2

    # Create input leaf modules with same scope
    scope = Scope(list(range(out_features)))
    leaf1 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)
    leaf2 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)

    # Create ElementwiseSum module
    module = ElementwiseSum(inputs=[leaf1, leaf2], out_channels=out_channels, num_repetitions=num_reps)

    # Get feature_to_scope from both modules
    feature_scopes = module.feature_to_scope
    first_input_scopes = module.inputs[0].feature_to_scope

    # Should delegate to first input's feature_to_scope
    assert np.array_equal(feature_scopes, first_input_scopes)

    # Validate shape matches first input
    assert feature_scopes.shape == (out_features, num_reps)

    # Validate all elements are Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    # Validate content matches first input (each feature should map to its corresponding scope)
    for f_idx in range(out_features):
        for r_idx in range(num_reps):
            assert feature_scopes[f_idx, r_idx] == first_input_scopes[f_idx, r_idx]
            # Each scope should contain the single feature (query is a tuple)
            assert feature_scopes[f_idx, r_idx].query == (f_idx,)


def test_feature_to_scope_multiple_inputs():
    """Test that feature_to_scope delegates to first input even with multiple inputs."""
    out_features = 5
    in_channels = 2
    out_channels = 3
    num_reps = 3

    # Create three input leaf modules with same scope
    scope = Scope(list(range(out_features)))
    leaf1 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)
    leaf2 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)
    leaf3 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)

    # Create ElementwiseSum module with three inputs
    module = ElementwiseSum(inputs=[leaf1, leaf2, leaf3], out_channels=out_channels, num_repetitions=num_reps)

    # Get feature_to_scope
    feature_scopes = module.feature_to_scope
    first_input_scopes = leaf1.feature_to_scope

    # Should delegate to first input's feature_to_scope
    assert np.array_equal(feature_scopes, first_input_scopes)

    # Validate shape
    assert feature_scopes.shape == (out_features, num_reps)

    # Validate all elements are Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    # Validate content (each feature should map to its scope)
    for f_idx in range(out_features):
        for r_idx in range(num_reps):
            assert feature_scopes[f_idx, r_idx].query == (f_idx,)


def test_feature_to_scope_with_product_inputs():
    """Test that feature_to_scope works correctly when inputs are product modules."""
    in_channels = 2
    out_channels = 3
    num_reps = 2

    # Create product modules for both inputs
    # Product joins scopes element-wise
    scope_a = Scope(list(range(0, 2)))
    scope_b = Scope(list(range(2, 4)))
    leaf_a1 = make_leaf(cls=DummyLeaf, out_channels=in_channels, scope=scope_a, num_repetitions=num_reps)
    leaf_b1 = make_leaf(cls=DummyLeaf, out_channels=in_channels, scope=scope_b, num_repetitions=num_reps)
    prod1 = ElementwiseProduct(inputs=[leaf_a1, leaf_b1])

    leaf_a2 = make_leaf(cls=DummyLeaf, out_channels=in_channels, scope=scope_a, num_repetitions=num_reps)
    leaf_b2 = make_leaf(cls=DummyLeaf, out_channels=in_channels, scope=scope_b, num_repetitions=num_reps)
    prod2 = ElementwiseProduct(inputs=[leaf_a2, leaf_b2])

    # Create ElementwiseSum module with product inputs
    module = ElementwiseSum(inputs=[prod1, prod2], out_channels=out_channels, num_repetitions=num_reps)

    # Get feature_to_scope
    feature_scopes = module.feature_to_scope

    # Should delegate to first product's feature_to_scope
    prod1_scopes = prod1.feature_to_scope
    assert np.array_equal(feature_scopes, prod1_scopes)

    # Product has same number of features as inputs (element-wise operation)
    expected_features = prod1.out_shape.features
    assert feature_scopes.shape == (expected_features, num_reps)

    # Validate all elements are Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    # Validate content - each feature should have a joined scope from both inputs
    for f_idx in range(expected_features):
        for r_idx in range(num_reps):
            # The product joins scopes, so each feature has scope from both leaf inputs
            assert len(feature_scopes[f_idx, r_idx].query) == 2  # Joined from 2 inputs


def test_constructor_requires_out_channels_or_weights():
    leaves = [
        make_normal_leaf(out_features=1, out_channels=1),
        make_normal_leaf(out_features=1, out_channels=1),
    ]
    with pytest.raises(ValueError):
        ElementwiseSum(inputs=leaves)


def test_constructor_rejects_feature_to_scope_mismatch_per_repetition():
    class _ToyModule(Module):
        def __init__(self, feature_to_scope: np.ndarray) -> None:
            super().__init__()
            self.scope = Scope([0, 1])
            self.in_shape = ModuleShape(features=2, channels=1, repetitions=2)
            self.out_shape = ModuleShape(features=2, channels=1, repetitions=2)
            self._feature_to_scope = feature_to_scope

        @property
        def feature_to_scope(self) -> np.ndarray:
            return self._feature_to_scope

        def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:  # pragma: no cover
            raise NotImplementedError

        def sample(
            self, num_samples=None, data=None, is_mpe: bool = False, cache=None, sampling_ctx=None
        ) -> torch.Tensor:  # pragma: no cover
            raise NotImplementedError

        def _sample(
            self,
            data: torch.Tensor,
            sampling_ctx: SamplingContext,
            cache: Cache,
        ) -> torch.Tensor:  # pragma: no cover
            raise NotImplementedError

        def marginalize(self, marg_rvs: list[int], prune: bool = True, cache=None):  # pragma: no cover
            raise NotImplementedError

    good = np.array([[Scope([0]), Scope([0])], [Scope([1]), Scope([1])]], dtype=object)
    bad = np.array([[Scope([0]), Scope([0, 1])], [Scope([1]), Scope([1])]], dtype=object)
    with pytest.raises(ScopeError):
        ElementwiseSum(inputs=[_ToyModule(good), _ToyModule(bad)], out_channels=1, num_repetitions=2)


def test_weights_setter_rejects_non_positive_values():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    invalid = module.weights.clone()
    invalid[0, 0, 0, 0, 0] = 0.0
    with pytest.raises(InvalidWeightsError):
        module.weights = invalid


def test_log_weights_setter_rejects_invalid_shape():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    with pytest.raises(ValueError):
        module.log_weights = torch.zeros((1,))


def test_extra_repr_contains_weights_shape():
    module = make_sum(in_channels=2, out_channels=3, out_features=2, num_repetitions=1)
    assert "weights=" in module.extra_repr()


def test_marginalize_with_non_overlapping_rvs_keeps_inputs():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    marg = module.marginalize([99], prune=True)
    assert marg is not None
    assert len(marg.inputs) == len(module.inputs)


def test_marginalize_returns_none_when_all_partially_marginalized_inputs_vanish(
    monkeypatch: pytest.MonkeyPatch,
):
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    for inp in module.inputs:
        monkeypatch.setattr(inp, "marginalize", lambda *args, **kwargs: None)
    assert module.marginalize([0], prune=True, cache=Cache()) is None


def test_sample_requires_repetition_index_for_multiple_repetitions():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=2)
    data = torch.full((4, 2), torch.nan)
    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((4, 2), dtype=torch.long),
        mask=torch.ones((4, 2), dtype=torch.bool),
        repetition_index=torch.zeros(4, dtype=torch.long),
    )
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == (4, 2)


def test_sample_mpe_path_runs():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    data = torch.full((4, 2), torch.nan)
    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((4, 2), dtype=torch.long),
        mask=torch.ones((4, 2), dtype=torch.bool),
        is_mpe=True,
    )
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    assert torch.isfinite(samples).all()
