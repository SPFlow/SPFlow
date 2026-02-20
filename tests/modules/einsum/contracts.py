"""Shared contract tests for Einsum-like layers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest
import torch

from spflow.learn import expectation_maximization
from spflow.meta import Scope
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import DummyLeaf, make_leaf, make_normal_data, make_normal_leaf


class EinsumLikeContractTests(ABC):
    """Shared behavior checks for Einsum-style layers."""

    __test__ = False

    in_channels_values = [1, 3]
    out_channels_values = [1, 4]
    in_features_values = [2, 4, 8]
    num_repetitions_values = [1, 2]

    @abstractmethod
    def layer_cls(self):
        """Return concrete layer class under test."""

    @abstractmethod
    def make_single_input(self, in_channels: int, out_channels: int, in_features: int, num_reps: int):
        """Build layer with one input branch."""

    @abstractmethod
    def make_two_inputs(self, in_channels: int, out_channels: int, in_features: int, num_reps: int):
        """Build layer with two explicit input branches."""

    @abstractmethod
    def expected_single_weight_shape(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ) -> tuple[int, ...]:
        """Return expected weight shape for single-input constructor."""

    @abstractmethod
    def assert_module_specific_two_input_channel_behavior(self) -> None:
        """Assert module-specific behavior for asymmetric two-input channels."""

    @abstractmethod
    def input_channel_reduce_dims(self) -> tuple[int, ...]:
        """Dimensions reduced when normalizing across input channels."""

    @pytest.mark.contract
    @pytest.mark.parametrize("in_channels", in_channels_values)
    @pytest.mark.parametrize("out_channels", out_channels_values)
    @pytest.mark.parametrize("in_features", in_features_values)
    @pytest.mark.parametrize("num_reps", num_repetitions_values)
    def test_single_input_construction_contract(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ) -> None:
        module = self.make_single_input(in_channels, out_channels, in_features, num_reps)

        assert module.out_shape.features == in_features // 2
        assert module.out_shape.channels == out_channels
        assert module.out_shape.repetitions == num_reps
        assert module.weights_shape == self.expected_single_weight_shape(
            in_channels=in_channels,
            out_channels=out_channels,
            in_features=in_features,
            num_reps=num_reps,
        )
        assert module.weights.shape == module.weights_shape

    @pytest.mark.contract
    @pytest.mark.parametrize("in_channels", in_channels_values)
    @pytest.mark.parametrize("out_channels", out_channels_values)
    @pytest.mark.parametrize("in_features", in_features_values)
    @pytest.mark.parametrize("num_reps", num_repetitions_values)
    def test_two_input_construction_contract(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ) -> None:
        module = self.make_two_inputs(in_channels, out_channels, in_features, num_reps)
        assert module.out_shape.features == in_features
        assert module.out_shape.channels == out_channels
        assert module.out_shape.repetitions == num_reps

    @pytest.mark.contract
    @pytest.mark.parametrize("in_channels", in_channels_values)
    @pytest.mark.parametrize("out_channels", out_channels_values)
    @pytest.mark.parametrize("in_features", in_features_values)
    @pytest.mark.parametrize("num_reps", num_repetitions_values)
    def test_log_likelihood_contract(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ) -> None:
        module = self.make_single_input(in_channels, out_channels, in_features, num_reps)
        data = make_normal_data(out_features=in_features)
        lls = module.log_likelihood(data)
        assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)
        assert torch.isfinite(lls).all()

    @pytest.mark.contract
    def test_cached_log_likelihood_contract(self) -> None:
        module = self.make_single_input(in_channels=2, out_channels=3, in_features=4, num_reps=1)
        data = make_normal_data(out_features=4)
        cache = Cache()

        lls = module.log_likelihood(data, cache=cache)

        assert "log_likelihood" in cache
        assert cache["log_likelihood"].get(module) is not None
        torch.testing.assert_close(cache["log_likelihood"][module], lls, rtol=0.0, atol=0.0)

    @pytest.mark.contract
    @pytest.mark.parametrize("num_reps", [1, 2])
    def test_sampling_contract(self, num_reps: int) -> None:
        module = self.make_single_input(in_channels=2, out_channels=3, in_features=4, num_reps=num_reps)
        n = 20
        data = torch.full((n, 4), torch.nan)
        channel_index = torch.randint(low=0, high=3, size=(n, module.out_shape.features))
        mask = torch.ones((n, module.out_shape.features), dtype=torch.bool)
        repetition_index = torch.randint(low=0, high=num_reps, size=(n,))
        ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
        samples = module._sample(data=data, sampling_ctx=ctx, cache=Cache())
        assert samples.shape == (n, 4)
        assert torch.isfinite(samples[:, module.scope.query]).all()

    @pytest.mark.contract
    @pytest.mark.parametrize("num_reps", [1, 2])
    def test_differentiable_sampling_contract(self, num_reps: int) -> None:
        module = self.make_single_input(in_channels=2, out_channels=3, in_features=4, num_reps=num_reps)
        n = 20
        data = torch.full((n, 4), torch.nan)
        int_channel_index = torch.randint(low=0, high=3, size=(n, module.out_shape.features))
        int_repetition_index = torch.randint(low=0, high=num_reps, size=(n,))
        mask = torch.ones((n, module.out_shape.features), dtype=torch.bool)
        ctx = SamplingContext(
            channel_index=to_one_hot(int_channel_index, dim=-1, dim_size=module.out_shape.channels),
            mask=mask,
            repetition_index=to_one_hot(int_repetition_index, dim=-1, dim_size=num_reps),
            is_differentiable=True,
        )
        samples = module._sample(data=data, sampling_ctx=ctx, cache=Cache())
        assert samples.shape == (n, 4)
        assert torch.isfinite(samples[:, module.scope.query]).all()

    @pytest.mark.contract
    def test_weights_are_normalized_contract(self) -> None:
        module = self.make_single_input(in_channels=2, out_channels=3, in_features=4, num_reps=1)
        sums = module.weights.sum(dim=self.input_channel_reduce_dims())
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-8)

    @pytest.mark.contract
    def test_em_updates_weights_contract(self) -> None:
        module = self.make_single_input(in_channels=2, out_channels=3, in_features=4, num_reps=1)
        data = make_normal_data(num_samples=30, out_features=4)
        before = module.weights.detach().clone()

        ll_history = expectation_maximization(module, data, max_steps=3)

        assert ll_history.ndim == 1
        assert ll_history.numel() >= 1
        assert torch.isfinite(ll_history).all()
        assert not torch.allclose(module.weights, before, rtol=0.0, atol=0.0)

    @pytest.mark.contract
    def test_invalid_single_feature_contract(self) -> None:
        inputs = make_normal_leaf(out_features=1, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            self.layer_cls()(inputs=inputs, out_channels=2)

    @pytest.mark.contract
    def test_two_input_channel_behavior_contract(self) -> None:
        self.assert_module_specific_two_input_channel_behavior()


def make_two_inputs_for_contract(
    *,
    in_channels: int,
    in_features: int,
    num_repetitions: int,
    left_channels: int | None = None,
    right_channels: int | None = None,
):
    """Create two disjoint-scope dummy inputs for shared contract tests."""
    left_ch = in_channels if left_channels is None else left_channels
    right_ch = in_channels if right_channels is None else right_channels
    left_scope = Scope(list(range(0, in_features)))
    right_scope = Scope(list(range(in_features, in_features * 2)))
    left = make_leaf(cls=DummyLeaf, out_channels=left_ch, scope=left_scope, num_repetitions=num_repetitions)
    right = make_leaf(
        cls=DummyLeaf, out_channels=right_ch, scope=right_scope, num_repetitions=num_repetitions
    )
    return left, right
