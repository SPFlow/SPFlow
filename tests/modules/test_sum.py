from itertools import product

import numpy as np
import pytest
import torch

from spflow.exceptions import (
    InvalidParameterCombinationError,
    InvalidWeightsError,
    MissingCacheError,
    ShapeError,
)
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.products import ElementwiseProduct
from spflow.modules.sums import Sum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_leaf, DummyLeaf

in_channels_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]
num_repetitions = [1, 7]
params = list(product(in_channels_values, out_channels_values, out_features_values, num_repetitions))


def _randn(*size: int) -> torch.Tensor:
    return torch.randn(*size)


def _rand(*size: int) -> torch.Tensor:
    return torch.rand(*size)


def _randint(low: int, high: int, size: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(low=low, high=high, size=size)


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

    # Only pass out_channels if explicitly specified, otherwise let Sum use default (1)
    # or infer from weights
    kwargs = {"inputs": inputs, "weights": weights, "num_repetitions": num_repetitions}
    if out_channels is not None:
        kwargs["out_channels"] = out_channels

    return Sum(**kwargs)


class _ToyInput(Module):
    def __init__(
        self, feature_to_scope: np.ndarray, out_channels: int = 2, return_none_on_marginalize: bool = False
    ):
        super().__init__()
        self._feature_to_scope = feature_to_scope
        self._return_none_on_marginalize = return_none_on_marginalize
        features, repetitions = feature_to_scope.shape
        self.in_shape = ModuleShape(features=features, channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=features, channels=out_channels, repetitions=repetitions)
        all_query_vars: set[int] = set()
        for scope in feature_to_scope.flatten():
            if scope is not None:
                all_query_vars.update(scope.query)
        self.scope = Scope(sorted(all_query_vars))

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self._feature_to_scope

    def log_likelihood(self, data, cache=None):
        return torch.zeros(
            data.shape[0],
            self.out_shape.features,
            self.out_shape.channels,
            self.out_shape.repetitions,
            device=data.device,
        )

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None):
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan"))
        data[:, self.scope.query] = 0.0
        return data

    def _sample(
        self,
        data: torch.Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> torch.Tensor:
        del sampling_ctx
        del cache
        del is_mpe
        data[:, self.scope.query] = 0.0
        return data

    def expectation_maximization(self, data, bias_correction: bool = True, cache: Cache | None = None):
        return None

    def maximum_likelihood_estimation(self, data, weights=None, cache: Cache | None = None):
        return None

    def marginalize(self, marg_rvs, prune: bool = True, cache: Cache | None = None):
        if self._return_none_on_marginalize:
            return None
        return self


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
    # Always expect 4D output
    assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)


@pytest.mark.parametrize(
    "in_channels,out_channels,out_features,num_reps",
    product(in_channels_values, out_channels_values, [2, 4], num_repetitions),
)
def test_log_likelihood_product_inputs(in_channels: int, out_channels: int, out_features: int, num_reps):
    inputs_a = make_leaf(
        cls=DummyLeaf,
        out_channels=in_channels,
        scope=Scope(range(0, out_features // 2)),
        num_repetitions=num_reps,
    )
    inputs_b = make_leaf(
        cls=DummyLeaf,
        out_channels=in_channels,
        scope=Scope(range(out_features // 2, out_features)),
        num_repetitions=num_reps,
    )
    inputs = [inputs_a, inputs_b]
    prod = ElementwiseProduct(inputs=inputs)

    module = Sum(out_channels=out_channels, inputs=prod, num_repetitions=num_reps)

    data = make_normal_data(out_features=out_features)
    lls = module.log_likelihood(data)
    assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_sample(in_channels: int, out_channels: int, out_features: int, num_reps):
    n_samples = 100
    module = make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    data = torch.full((n_samples, module.out_shape.features), torch.nan)
    channel_index = _randint(
        low=0, high=module.out_shape.channels, size=(n_samples, module.out_shape.features)
    )
    mask = torch.full((n_samples, module.out_shape.features), True)
    repetition_index = _randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = module.sample(data=data)
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
        cls=DummyLeaf,
        out_channels=in_channels,
        scope=Scope(range(0, out_features // 2)),
        num_repetitions=num_reps,
    )
    inputs_b = make_leaf(
        cls=DummyLeaf,
        out_channels=in_channels,
        scope=Scope(range(out_features // 2, out_features)),
        num_repetitions=num_reps,
    )
    inputs = [inputs_a, inputs_b]
    prod = ElementwiseProduct(inputs=inputs)

    module = Sum(out_channels=out_channels, inputs=prod, num_repetitions=num_reps)

    data = torch.full((n_samples, out_features), torch.nan)
    channel_index = _randint(
        low=0, high=module.out_shape.channels, size=(n_samples, module.out_shape.features)
    )
    mask = torch.full((n_samples, module.out_shape.features), True)
    repetition_index = _randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = module.sample(data=data)
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

    # Create some data
    data = _randn(n_samples, module.out_shape.features)

    # Set first three scopes to nan
    data[:, [0, 1, 2]] = torch.nan

    data_copy = data.clone()

    cache = Cache()
    samples = module.sample_with_evidence(
        evidence=data,
        is_mpe=False,
        cache=cache,
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
    torch.testing.assert_close(data_copy[:, [3, 4, 5]], samples[:, [3, 4, 5]], rtol=0.0, atol=0.0)


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
    loc_before = module.inputs.loc.detach().clone()
    scale_before = module.inputs.scale.detach().clone()

    max_steps = 2
    ll_history = expectation_maximization(module, data, max_steps=max_steps)
    assert ll_history.ndim == 1
    assert 1 <= ll_history.numel() <= max_steps
    assert ll_history.isfinite().all()

    assert not torch.equal(module.inputs.loc, loc_before)
    assert not torch.equal(module.inputs.scale, scale_before)
    torch.testing.assert_close(module.inputs.loc, torch.zeros_like(module.inputs.loc))
    torch.testing.assert_close(module.inputs.scale, torch.ones_like(module.inputs.scale))


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
        assert not torch.allclose(module.weights, weights_before, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("in_channels,out_channels,out_features, num_reps", params)
def test_weights(in_channels: int, out_channels: int, out_features: int, num_reps):
    weights = torch.ones((out_features, in_channels, out_channels, num_reps))
    weights = weights / weights.sum(dim=1, keepdim=True)

    module = make_sum(
        weights=weights,
        in_channels=in_channels,
    )
    weights_sum = module.weights.sum(dim=module.sum_dim)
    torch.testing.assert_close(weights_sum, torch.ones_like(weights_sum), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(module.log_weights, module.weights.log(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", params)
def test_invalid_weights_not_normalized(in_channels: int, out_channels: int, out_features: int, num_reps):
    weights = _rand((out_features, in_channels, out_channels, num_reps))
    with pytest.raises(InvalidWeightsError):
        make_sum(weights=weights, in_channels=in_channels)


@pytest.mark.parametrize("in_channels,out_channels,out_features, num_reps", params)
def test_invalid_weights_negative(in_channels: int, out_channels: int, out_features: int, num_reps):
    weights = _rand((out_features, in_channels, out_channels, num_reps)) - 1.0
    with pytest.raises(InvalidWeightsError):
        make_sum(weights=weights, in_channels=in_channels)


def test_invalid_weights_wrong_dim():
    weights = _rand((2, 2, 2, 2, 2))
    with pytest.raises(ShapeError):
        make_sum(weights=weights, in_channels=2)


def test_invalid_out_channels_and_weights():
    weights = torch.ones((2, 2, 2, 1))
    weights /= weights.sum(dim=1, keepdim=True)
    inputs = make_normal_leaf(out_features=2, out_channels=2, num_repetitions=1)
    with pytest.raises(InvalidParameterCombinationError):
        Sum(out_channels=2, inputs=inputs, weights=weights)


def test_expectation_maximization_cache_error():
    """Test that expectation_maximization raises ValueError when cache is missing entries."""
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    data = _randn(5, 2)
    cache = Cache()
    # Missing log_likelihood in cache
    with pytest.raises(MissingCacheError):
        module._expectation_maximization_step(data, cache=cache)


def test_constructor_empty_inputs():
    """Test that Sum raises ValueError when inputs is empty."""
    with pytest.raises(ValueError):
        Sum(inputs=[], out_channels=2)


def test_constructor_invalid_out_channels():
    """Test that Sum raises ValueError when out_channels < 1."""
    leaf = make_normal_leaf(scope=Scope([0]), out_channels=2)
    with pytest.raises(ValueError):
        Sum(inputs=[leaf], out_channels=0)


def test_constructor_sets_default_repetitions_when_none():
    leaf = make_normal_leaf(scope=Scope([0, 1]), out_channels=2, num_repetitions=1)
    module = Sum(inputs=leaf, out_channels=2, num_repetitions=None)
    assert module.out_shape.repetitions == 1


def test_constructor_list_with_single_input_keeps_underlying_module():
    leaf = make_normal_leaf(scope=Scope([0, 1]), out_channels=2, num_repetitions=1)
    module = Sum(inputs=[leaf], out_channels=3)
    assert module.inputs is leaf


def test_weights_can_be_initialized_from_2d_tensor():
    weights = torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.float32)
    leaf = make_normal_leaf(scope=Scope([0]), out_channels=2, num_repetitions=1)
    module = Sum(inputs=leaf, weights=weights)
    assert module.weights_shape == (1, 2, 2, 1)


def test_weights_can_be_initialized_from_1d_tensor():
    weights = torch.tensor([0.2, 0.8], dtype=torch.float32)
    leaf = make_normal_leaf(scope=Scope([0]), out_channels=2, num_repetitions=1)
    module = Sum(inputs=leaf, weights=weights)
    assert module.weights_shape == (1, 2, 1, 1)


def test_weights_can_be_initialized_from_3d_tensor():
    weights = torch.tensor(
        [
            [[0.2, 0.8], [0.8, 0.2]],
            [[0.6, 0.4], [0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    leaf = make_normal_leaf(scope=Scope([0, 1]), out_channels=2, num_repetitions=1)
    module = Sum(inputs=leaf, weights=weights)
    assert module.weights_shape == (2, 2, 2, 1)


def test_weights_can_be_initialized_from_4d_tensor():
    weights = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    weights /= weights.sum(dim=1, keepdim=True)
    leaf = make_normal_leaf(scope=Scope([0]), out_channels=2, num_repetitions=2)
    module = Sum(inputs=leaf, weights=weights)
    assert module.weights_shape == (1, 2, 2, 2)
    assert module.out_shape.repetitions == 2


def test_setting_weights_with_invalid_shape_raises():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    with pytest.raises(ShapeError):
        module.weights = torch.ones((2, 2, 2))


def test_setting_log_weights_with_invalid_shape_raises():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    with pytest.raises(ShapeError):
        module.log_weights = torch.zeros((2, 2, 2))


def test_log_likelihood_creates_default_cache():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    data = make_normal_data(out_features=2, num_samples=4)
    lls = module.log_likelihood(data, cache=None)
    assert lls.shape == (4, 2, 2, 1)


def test_sample_bootstraps_root_sampling_context():
    module = make_sum(in_channels=2, out_channels=2, out_features=3, num_repetitions=1)
    samples = module.sample()
    assert samples.shape == (1, 3)


def test_sample_defaults_repetition_index_for_multiple_repetitions():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=2)
    samples = module.sample(num_samples=3)
    assert samples.shape == (3, 2)


def test_sample_raises_on_incompatible_mask_width():
    module = make_sum(in_channels=2, out_channels=2, out_features=3, num_repetitions=1)

    class _LooseSamplingCtx:
        def __init__(self):
            self.channel_index = torch.zeros((5, 3), dtype=torch.long)
            self.mask = torch.ones((5, 1), dtype=torch.bool)
            self.repetition_idx = None

        def update(self, channel_index, mask):
            self.channel_index = channel_index
            self.mask = mask

        def require_feature_width(self, expected_features):
            if self.channel_index.shape[1] != expected_features:
                raise ShapeError(
                    "Received incompatible sampling context feature width: "
                    f"got {self.channel_index.shape[1]}, expected {expected_features}."
                )
            if self.mask.shape[1] != expected_features:
                raise ShapeError(
                    "Received incompatible sampling context feature width: "
                    f"got {self.mask.shape[1]}, expected {expected_features}."
                )

    sampling_ctx = _LooseSamplingCtx()
    with pytest.raises(ShapeError, match="incompatible sampling context feature width"):
        module._sample(data=torch.full((5, 3), torch.nan), sampling_ctx=sampling_ctx, cache=Cache())


def test_expectation_maximization_requires_cached_lls():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    cache = Cache()
    with pytest.raises(MissingCacheError):
        module._expectation_maximization_step(_randn(3, 2), cache=cache)


def test_expectation_maximization_raises_for_missing_module_lls():
    module = make_sum(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    cache = Cache()
    cache["log_likelihood"][module.inputs] = torch.zeros(3, 2, 2, 1)
    with pytest.raises(MissingCacheError):
        module._expectation_maximization_step(_randn(3, 2), cache=cache)


def test_marginalize_handles_different_features_per_repetition():
    feature_to_scope = np.array(
        [
            [Scope([0]), Scope([0])],
            [Scope([1]), Scope([0, 1])],
        ],
        dtype=object,
    )
    inputs = _ToyInput(feature_to_scope=feature_to_scope, out_channels=2)
    module = Sum(inputs=inputs, out_channels=2, num_repetitions=2)
    with pytest.raises(InvalidWeightsError):
        module.marginalize([1])


def test_marginalize_returns_none_when_input_marginalization_returns_none():
    feature_to_scope = np.array([[Scope([0]), Scope([0])], [Scope([1]), Scope([1])]], dtype=object)
    inputs = _ToyInput(
        feature_to_scope=feature_to_scope,
        out_channels=2,
        return_none_on_marginalize=True,
    )
    module = Sum(inputs=inputs, out_channels=2)
    assert module.marginalize([0]) is None


def test_num_repetitions_mismatch_with_weights():
    """Test InvalidParameterCombinationError when num_repetitions mismatches weights shape."""
    out_features, in_channels, out_channels = 2, 2, 2
    # weights with 3 repetitions
    weights = torch.ones((out_features, in_channels, out_channels, 3))
    weights /= weights.sum(dim=1, keepdim=True)
    leaf = make_normal_leaf(scope=Scope([0, 1]), out_channels=in_channels, num_repetitions=3)

    # Passing num_repetitions=2 which mismatches weights shape (3)
    with pytest.raises(InvalidParameterCombinationError):
        Sum(inputs=leaf, weights=weights, num_repetitions=2)


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

    mean = _rand((out_features, out_channels, num_reps))
    std = _rand((out_features, out_channels, num_reps))

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

    module_b = Sum(inputs=[normal_layer_b1, normal_layer_b2], weights=module_a.weights)

    # test log likelihood

    data = make_normal_data(out_features=out_features)

    ll_a = module_a.log_likelihood(data)
    ll_b = module_b.log_likelihood(data)

    torch.testing.assert_close(ll_a, ll_b, rtol=1e-5, atol=1e-6)

    # test sampling

    n_samples = 10

    data_a = torch.full((n_samples, out_features), torch.nan)
    channel_index = _randint(low=0, high=sum_out_channels, size=(n_samples, out_features))
    mask = torch.full((n_samples, out_features), True)
    repetition_index = _randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index, mask=mask, repetition_index=repetition_index
    )

    data_b = torch.full((n_samples, out_features), torch.nan)

    sampling_ctx_b = SamplingContext(
        channel_index=channel_index, mask=mask, repetition_index=repetition_index
    )

    samples_a = module_a.sample(data=data_a, is_mpe=True)
    samples_b = module_b.sample(data=data_b, is_mpe=True)

    torch.testing.assert_close(samples_a, samples_b, rtol=0.0, atol=0.0)


def test_feature_to_scope_single_input():
    """Test that feature_to_scope correctly delegates to input module with single input."""
    out_features = 6
    in_channels = 3
    out_channels = 4
    num_reps = 2

    # Create input leaf module
    scope = Scope(list(range(out_features)))
    leaf = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)

    # Create Sum module with single input
    module = Sum(inputs=leaf, out_channels=out_channels, num_repetitions=num_reps)

    # Get feature_to_scope from both modules
    feature_scopes = module.feature_to_scope
    leaf_scopes = leaf.feature_to_scope

    # Should delegate to input's feature_to_scope
    assert np.array_equal(feature_scopes, leaf_scopes)

    # Validate shape matches input
    assert feature_scopes.shape == (out_features, num_reps)

    # Validate all elements are Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    # Validate content matches input (each feature should map to its corresponding scope)
    for f_idx in range(out_features):
        for r_idx in range(num_reps):
            assert feature_scopes[f_idx, r_idx] == leaf_scopes[f_idx, r_idx]
            # Each scope should contain the single feature (query is a tuple)
            assert feature_scopes[f_idx, r_idx].query == (f_idx,)


def test_feature_to_scope_multiple_inputs():
    """Test that feature_to_scope correctly delegates to Cat module with multiple inputs."""
    out_features = 4
    in_channels = 2
    out_channels = 3
    num_reps = 3

    # Create two input leaf modules with same scope
    scope = Scope(list(range(out_features)))
    leaf1 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)
    leaf2 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)

    # Create Sum module with multiple inputs (internally creates Cat)
    module = Sum(inputs=[leaf1, leaf2], out_channels=out_channels, num_repetitions=num_reps)

    # Get feature_to_scope
    feature_scopes = module.feature_to_scope

    # Should delegate to Cat input's feature_to_scope
    cat_scopes = module.inputs.feature_to_scope
    assert np.array_equal(feature_scopes, cat_scopes)

    # Validate shape
    assert feature_scopes.shape == (out_features, num_reps)

    # Validate all elements are Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    # Validate content (each feature should map to its scope)
    for f_idx in range(out_features):
        for r_idx in range(num_reps):
            assert feature_scopes[f_idx, r_idx].query == (f_idx,)


def test_feature_to_scope_with_product_input():
    """Test that feature_to_scope works correctly when input is a product module."""
    in_channels = 2
    out_channels = 3
    num_reps = 2

    # Create two leaf modules for different scopes
    # Product will have features equal to the number of features in the first input
    # since it operates element-wise
    scope_a = Scope(list(range(0, 2)))
    scope_b = Scope(list(range(2, 4)))
    leaf_a = make_leaf(cls=DummyLeaf, out_channels=in_channels, scope=scope_a, num_repetitions=num_reps)
    leaf_b = make_leaf(cls=DummyLeaf, out_channels=in_channels, scope=scope_b, num_repetitions=num_reps)

    # Create product module - this joins scopes element-wise
    prod = ElementwiseProduct(inputs=[leaf_a, leaf_b])

    # Create Sum module with product input
    module = Sum(inputs=prod, out_channels=out_channels, num_repetitions=num_reps)

    # Get feature_to_scope
    feature_scopes = module.feature_to_scope

    # Should delegate to product's feature_to_scope
    prod_scopes = prod.feature_to_scope
    assert np.array_equal(feature_scopes, prod_scopes)

    # Product has same number of features as inputs (element-wise operation)
    expected_features = prod.out_shape.features
    assert feature_scopes.shape == (expected_features, num_reps)

    # Validate all elements are Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    # Validate content - each feature should have a joined scope from both inputs
    for f_idx in range(expected_features):
        for r_idx in range(num_reps):
            # The product joins scopes, so each feature has scope from both leaf_a and leaf_b
            assert len(feature_scopes[f_idx, r_idx].query) == 2  # Joined from 2 inputs


@pytest.mark.parametrize("num_reps", [1, 3, 7])
def test_feature_to_scope_with_repetitions(num_reps: int):
    """Test that feature_to_scope correctly handles different num_repetitions values."""
    out_features = 5
    in_channels = 2
    out_channels = 3

    # Create input leaf module with specified repetitions
    scope = Scope(list(range(out_features)))
    leaf = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)

    # Create Sum module
    module = Sum(inputs=leaf, out_channels=out_channels, num_repetitions=num_reps)

    # Get feature_to_scope
    feature_scopes = module.feature_to_scope

    # Should delegate to input's feature_to_scope
    leaf_scopes = leaf.feature_to_scope
    assert np.array_equal(feature_scopes, leaf_scopes)

    # Validate shape includes repetitions
    assert feature_scopes.shape == (out_features, num_reps)

    # Validate all elements are Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    # Validate each repetition has correct scopes
    for r_idx in range(num_reps):
        for f_idx in range(out_features):
            assert feature_scopes[f_idx, r_idx].query == (f_idx,)
            # All repetitions should have same scope structure (may differ in content for certain modules)
            assert feature_scopes[f_idx, r_idx] == leaf_scopes[f_idx, r_idx]
