from itertools import product

import numpy as np
import pytest
import torch

from spflow.exceptions import (
    InvalidParameterCombinationError,
    InvalidParameterError,
    InvalidWeightsError,
    MissingCacheError,
    ShapeError,
)
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.meta import Scope
from spflow.modules.leaves import Bernoulli
from spflow.modules.leaves import Normal
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.products import ElementwiseProduct
from spflow.modules.sums import Sum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_leaf, DummyLeaf
from tests.utils.sampling_context_helpers import patch_simple_as_categorical_one_hot

in_channels_values = [1, 4]
out_channels_values = [1, 5]
num_repetitions = [1, 7]


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

    # Avoid forcing constructor conflicts when weights already determine channel count.
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
    ) -> torch.Tensor:
        del sampling_ctx
        del cache
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


# Cross-module behavioral contracts for the sum family moved to:
# - test_sum_contract_loglikelihood.py
# - test_sum_contract_sampling.py
# - test_sum_contract_training.py
# - test_sum_contract_marginalization.py
# - test_sum_contract_weights_and_constructor.py


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
    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((5, 3), dtype=torch.long),
        mask=torch.ones((5, 3), dtype=torch.bool),
    )
    sampling_ctx._mask = torch.ones((5, 1), dtype=torch.bool)  # type: ignore[attr-defined]
    with pytest.raises(InvalidParameterError, match="mismatched channel_index/mask shapes"):
        module._sample(data=torch.full((5, 3), torch.nan), sampling_ctx=sampling_ctx, cache=Cache())


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
    weights = torch.ones((out_features, in_channels, out_channels, 3))
    weights /= weights.sum(dim=1, keepdim=True)
    leaf = make_normal_leaf(scope=Scope([0, 1]), out_channels=in_channels, num_repetitions=3)

    # Explicit repetitions must agree with weight tensor metadata.
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

    marginalized_module = module.marginalize(marg_rvs, prune=prune)

    if len(marg_rvs) == out_features:
        assert marginalized_module is None
        return

    # Marginalized variables must be removed from the resulting scope.
    assert len(set(marginalized_module.scope.query).intersection(marg_rvs)) == 0

    # Only the feature axis should shrink after dropping marginalized variables.
    assert marginalized_module.weights.shape[0] == weights_shape[0] - len(marg_rvs)

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

    # Lock equivalence between concatenated-child and pre-concatenated constructions.

    data = make_normal_data(out_features=out_features)

    ll_a = module_a.log_likelihood(data)
    ll_b = module_b.log_likelihood(data)

    torch.testing.assert_close(ll_a, ll_b, rtol=1e-5, atol=1e-6)

    # Shared routing must produce identical samples across equivalent graph layouts.

    n_samples = 10

    data_a = torch.full((n_samples, out_features), torch.nan)
    channel_index = _randint(low=0, high=sum_out_channels, size=(n_samples, out_features))
    mask = torch.full((n_samples, out_features), True)
    repetition_index = _randint(low=0, high=num_reps, size=(n_samples,))
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


def test_feature_to_scope_single_input():
    """Test that feature_to_scope correctly delegates to input module with single input."""
    out_features = 6
    in_channels = 3
    out_channels = 4
    num_reps = 2

    scope = Scope(list(range(out_features)))
    leaf = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)

    module = Sum(inputs=leaf, out_channels=out_channels, num_repetitions=num_reps)

    feature_scopes = module.feature_to_scope
    leaf_scopes = leaf.feature_to_scope

    # Single-input Sum should be transparent in scope mapping.
    assert np.array_equal(feature_scopes, leaf_scopes)

    assert feature_scopes.shape == (out_features, num_reps)

    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    for f_idx in range(out_features):
        for r_idx in range(num_reps):
            assert feature_scopes[f_idx, r_idx] == leaf_scopes[f_idx, r_idx]
            assert feature_scopes[f_idx, r_idx].query == (f_idx,)


def test_feature_to_scope_multiple_inputs():
    """Test that feature_to_scope correctly delegates to Cat module with multiple inputs."""
    out_features = 4
    in_channels = 2
    out_channels = 3
    num_reps = 3

    scope = Scope(list(range(out_features)))
    leaf1 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)
    leaf2 = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)

    module = Sum(inputs=[leaf1, leaf2], out_channels=out_channels, num_repetitions=num_reps)

    feature_scopes = module.feature_to_scope

    # Multi-input Sum should expose the Cat-composed scope map unchanged.
    cat_scopes = module.inputs.feature_to_scope
    assert np.array_equal(feature_scopes, cat_scopes)

    assert feature_scopes.shape == (out_features, num_reps)

    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    for f_idx in range(out_features):
        for r_idx in range(num_reps):
            assert feature_scopes[f_idx, r_idx].query == (f_idx,)


def test_feature_to_scope_with_product_input():
    """Test that feature_to_scope works correctly when input is a product module."""
    in_channels = 2
    out_channels = 3
    num_reps = 2

    # Distinct child scopes make scope-joining behavior observable.
    scope_a = Scope(list(range(0, 2)))
    scope_b = Scope(list(range(2, 4)))
    leaf_a = make_leaf(cls=DummyLeaf, out_channels=in_channels, scope=scope_a, num_repetitions=num_reps)
    leaf_b = make_leaf(cls=DummyLeaf, out_channels=in_channels, scope=scope_b, num_repetitions=num_reps)

    prod = ElementwiseProduct(inputs=[leaf_a, leaf_b])

    module = Sum(inputs=prod, out_channels=out_channels, num_repetitions=num_reps)

    feature_scopes = module.feature_to_scope

    # Sum should preserve upstream scope aggregation from product inputs.
    prod_scopes = prod.feature_to_scope
    assert np.array_equal(feature_scopes, prod_scopes)

    expected_features = prod.out_shape.features
    assert feature_scopes.shape == (expected_features, num_reps)

    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    for f_idx in range(expected_features):
        for r_idx in range(num_reps):
            assert len(feature_scopes[f_idx, r_idx].query) == 2  # Confirms join includes both child scopes.


@pytest.mark.parametrize("num_reps", [1, 3, 7])
def test_feature_to_scope_with_repetitions(num_reps: int):
    """Test that feature_to_scope correctly handles different num_repetitions values."""
    out_features = 5
    in_channels = 2
    out_channels = 3

    scope = Scope(list(range(out_features)))
    leaf = make_normal_leaf(scope=scope, out_channels=in_channels, num_repetitions=num_reps)

    module = Sum(inputs=leaf, out_channels=out_channels, num_repetitions=num_reps)

    feature_scopes = module.feature_to_scope

    # Repetition count should not break transparent scope delegation.
    leaf_scopes = leaf.feature_to_scope
    assert np.array_equal(feature_scopes, leaf_scopes)

    assert feature_scopes.shape == (out_features, num_reps)

    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())

    for r_idx in range(num_reps):
        for f_idx in range(out_features):
            assert feature_scopes[f_idx, r_idx].query == (f_idx,)
            assert feature_scopes[f_idx, r_idx] == leaf_scopes[f_idx, r_idx]


class TestDifferentiableSampling:
    def test_diff_sampling(self):
        torch.manual_seed(7)
        n_samples = 32
        out_features = 1
        in_channels = 4
        sum_out_channels = 3

        leaf_a = Normal(scope=0, out_channels=in_channels, num_repetitions=1)
        leaf_b = Normal(scope=0, out_channels=in_channels, num_repetitions=1)
        module = Sum(inputs=[leaf_a, leaf_b], out_channels=sum_out_channels, num_repetitions=1)

        channel_index = torch.zeros((n_samples, out_features), dtype=torch.long)
        channel_index = to_one_hot(channel_index, dim=-1, dim_size=sum_out_channels)
        repetition_index = torch.zeros((n_samples,), dtype=torch.long)
        repetition_index = to_one_hot(repetition_index, dim=-1, dim_size=1)
        sampling_ctx = SamplingContext(
            channel_index=channel_index,
            mask=torch.full((n_samples, out_features), True),
            repetition_index=repetition_index,
            is_differentiable=True,
            hard=True,
            tau=1.0,
        )

        # Make `data` require grad (and be non-leaf) so in-place sampling writes
        # are tracked via autograd's CopySlices.
        data = torch.full((n_samples, out_features), torch.nan, requires_grad=True)
        data = data + 0.0

        out = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
        assert out.requires_grad
        assert torch.isfinite(out).all()

        loss = out.square().mean()
        loss.backward()

        assert module.logits.grad is not None
        assert torch.isfinite(module.logits.grad).all()
        assert float(module.logits.grad.abs().sum()) > 0.0

        for leaf in (leaf_a, leaf_b):
            assert leaf.loc.grad is not None
            assert leaf.log_scale.grad is not None
            assert torch.isfinite(leaf.loc.grad).all()
            assert torch.isfinite(leaf.log_scale.grad).all()
            assert float(leaf.loc.grad.abs().sum()) > 0.0
            assert float(leaf.log_scale.grad.abs().sum()) > 0.0

    def test_diff_sampling_smoke_with_discrete_leaf(self):
        torch.manual_seed(0)

        leaf = Bernoulli(scope=Scope([0]), out_channels=3, num_repetitions=1)
        weights = torch.tensor([[[[0.2]], [[0.3]], [[0.5]]]], dtype=torch.get_default_dtype())
        module = Sum(inputs=leaf, weights=weights)

        n_samples = 32
        data = torch.full((n_samples, 1), float("nan"))
        sampling_ctx = SamplingContext(
            num_samples=n_samples,
            device=data.device,
            is_differentiable=True,
            hard=True,
            tau=0.7,
        )
        out = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

        assert out.shape == (n_samples, 1)
        assert torch.isfinite(out).all()
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_diff_sampling_equals_non_diff_sampling(self, monkeypatch: pytest.MonkeyPatch):
        in_channels = 2
        out_features = 4
        num_reps = 5
        sum_out_channels = 3
        n_samples = 10

        feature_to_scope = np.array(
            [[Scope([i]) for _ in range(num_reps)] for i in range(out_features)],
            dtype=object,
        )
        toy_input = _ToyInput(feature_to_scope=feature_to_scope, out_channels=in_channels)
        module = Sum(inputs=toy_input, out_channels=sum_out_channels, num_repetitions=num_reps)

        weights = torch.arange(
            1,
            1 + out_features * in_channels * sum_out_channels * num_reps,
            dtype=torch.get_default_dtype(),
        ).reshape(out_features, in_channels, sum_out_channels, num_reps)
        weights /= weights.sum(dim=1, keepdim=True)
        module.weights = weights

        channel_index = (
            torch.arange(n_samples * out_features).reshape(n_samples, out_features) % sum_out_channels
        )
        repetition_index = torch.arange(n_samples) % num_reps
        mask = torch.full((n_samples, out_features), True)

        sampling_ctx_a = SamplingContext(
            channel_index=channel_index.clone(),
            mask=mask.clone(),
            repetition_index=repetition_index.clone(),
            is_mpe=False,
        )
        sampling_ctx_b = SamplingContext(
            channel_index=to_one_hot(channel_index, dim=-1, dim_size=sum_out_channels),
            mask=mask.clone(),
            repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
            is_mpe=False,
            is_differentiable=True,
        )

        patch_simple_as_categorical_one_hot(monkeypatch)

        torch.manual_seed(1337)
        samples_a = module._sample(
            data=torch.full((n_samples, out_features), torch.nan),
            sampling_ctx=sampling_ctx_a,
            cache=Cache(),
        )
        torch.manual_seed(1337)
        samples_b = module._sample(
            data=torch.full((n_samples, out_features), torch.nan),
            sampling_ctx=sampling_ctx_b,
            cache=Cache(),
        )

        torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            sampling_ctx_b.channel_index,
            to_one_hot(sampling_ctx_a.channel_index, dim=-1, dim_size=in_channels),
            rtol=0.0,
            atol=0.0,
        )

    def test_diff_sampling_with_conditional_cache(self):
        in_channels = 2
        out_features = 4
        num_reps = 3
        sum_out_channels = 3
        n_samples = 10

        normal_layer = make_normal_leaf(
            out_features=out_features,
            out_channels=in_channels,
            num_repetitions=num_reps,
        )
        module = Sum(inputs=normal_layer, out_channels=sum_out_channels, num_repetitions=num_reps)

        cache = Cache()
        evidence = _randn(n_samples, out_features)
        _ = module.log_likelihood(evidence, cache=cache)

        channel_index = _randint(low=0, high=sum_out_channels, size=(n_samples, out_features))
        repetition_index = _randint(low=0, high=num_reps, size=(n_samples,))
        sampling_ctx = SamplingContext(
            channel_index=to_one_hot(channel_index, dim=-1, dim_size=sum_out_channels),
            mask=torch.full((n_samples, out_features), True),
            repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
            is_differentiable=True,
        )

        samples = module._sample(
            data=torch.full((n_samples, out_features), torch.nan),
            sampling_ctx=sampling_ctx,
            cache=cache,
        )

        assert samples.shape == (n_samples, out_features)
        assert torch.isfinite(samples).all()
