"""Training and branch coverage tests for LinsumLayer."""

from itertools import product

import pytest
import torch

from spflow.learn import expectation_maximization
from spflow.meta import Scope
from spflow.modules.einsum import LinsumLayer
from spflow.modules.ops.split import SplitMode
from spflow.modules.ops.split_interleaved import SplitInterleaved
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.modules.einsum.layer_test_utils import make_linsum_single_input, make_linsum_two_inputs
from tests.utils.leaves import CachingDummyInput, DummyLeaf, make_leaf, make_normal_data, make_normal_leaf


class TestLinsumLayerGradientDescent:
    def test_gradient_flow(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        data = make_normal_data(out_features=4, num_samples=20)

        lls = module.log_likelihood(data)
        loss = -lls.mean()
        loss.backward()

        assert module.logits.grad is not None
        assert torch.isfinite(module.logits.grad).all()

    def test_optimization_changes_weights(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        data = make_normal_data(out_features=4, num_samples=50)

        weights_before = module.weights.clone()

        optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
        for _ in range(5):
            optimizer.zero_grad()
            lls = module.log_likelihood(data)
            loss = -lls.mean()
            loss.backward()
            optimizer.step()

        assert not torch.allclose(module.weights, weights_before, rtol=0.0, atol=0.0)


class TestLinsumLayerExpectationMaximization:
    @pytest.mark.parametrize(
        "in_channels,out_channels,in_features,num_reps",
        product([2], [3], [4], [1, 2]),
    )
    def test_em_weights_normalized_after_update_single_input(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        module = make_linsum_single_input(in_channels, out_channels, in_features, num_reps)
        data = make_normal_data(out_features=in_features, num_samples=50)

        expectation_maximization(module, data, max_steps=3)

        weights = module.weights
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)


class TestLinsumLayerEMTwoInputs:
    def make_linsum_with_caching_inputs(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ) -> LinsumLayer:
        left_scope = Scope(list(range(0, in_features)))
        right_scope = Scope(list(range(in_features, in_features * 2)))

        left_input = CachingDummyInput(
            out_channels=in_channels,
            num_repetitions=num_reps,
            out_features=in_features,
            scope=left_scope,
        )
        right_input = CachingDummyInput(
            out_channels=in_channels,
            num_repetitions=num_reps,
            out_features=in_features,
            scope=right_scope,
        )

        return LinsumLayer(
            inputs=[left_input, right_input], out_channels=out_channels, num_repetitions=num_reps
        )

    @pytest.mark.parametrize(
        "in_channels,out_channels,in_features,num_reps",
        product([2], [3], [4], [1, 2]),
    )
    def test_em_updates_weights_two_inputs(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        module = self.make_linsum_with_caching_inputs(in_channels, out_channels, in_features, num_reps)
        total_features = in_features * 2
        data = torch.randn(50, total_features)

        original_weights = module.weights.clone()

        ll_history = expectation_maximization(module, data, max_steps=5)

        assert not torch.allclose(module.weights, original_weights, rtol=0.0, atol=0.0)
        assert len(ll_history) >= 1
        assert torch.isfinite(ll_history).all()

    @pytest.mark.parametrize(
        "in_channels,out_channels,in_features,num_reps",
        product([2], [3], [4], [1, 2]),
    )
    def test_em_weights_normalized_two_inputs(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        module = self.make_linsum_with_caching_inputs(in_channels, out_channels, in_features, num_reps)
        total_features = in_features * 2
        data = torch.randn(50, total_features)

        expectation_maximization(module, data, max_steps=3)

        weights = module.weights
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)

    def test_em_raises_without_cache(self):
        module = self.make_linsum_with_caching_inputs(2, 3, 4, 1)
        data = torch.randn(10, 8)
        cache = Cache()

        with pytest.raises(ValueError):
            module.expectation_maximization(data, cache=cache)

    def test_two_inputs_flag_set_correctly(self):
        single_module = make_linsum_single_input(2, 3, 4, 1)
        assert not single_module._two_inputs

        two_input_module = self.make_linsum_with_caching_inputs(2, 3, 4, 1)
        assert two_input_module._two_inputs


class TestLinsumLayerCoverageBranches:
    def test_invalid_input_list_length_raises(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            LinsumLayer(inputs=[leaf], out_channels=2)

    def test_invalid_two_inputs_feature_mismatch_raises(self):
        left = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0, 1]), num_repetitions=1)
        right = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([2, 3, 4]), num_repetitions=1)
        with pytest.raises(ValueError):
            LinsumLayer(inputs=[left, right], out_channels=2)

    def test_invalid_two_inputs_repetition_mismatch_raises(self):
        left = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0, 1]), num_repetitions=1)
        right = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([2, 3]), num_repetitions=2)
        with pytest.raises(ValueError):
            LinsumLayer(inputs=[left, right], out_channels=2)

    def test_two_input_num_repetitions_inferred_when_none(self):
        left = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0, 1]), num_repetitions=3)
        right = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([2, 3]), num_repetitions=3)
        module = LinsumLayer(inputs=[left, right], out_channels=2, num_repetitions=None)
        assert module.out_shape.repetitions == 3

    def test_invalid_out_channels_raises(self):
        inputs = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            LinsumLayer(inputs=inputs, out_channels=0)

    def test_invalid_init_weight_shape_raises(self):
        inputs = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            LinsumLayer(inputs=inputs, out_channels=3, weights=torch.rand(1, 2, 3))

    def test_set_non_positive_weights_raises(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        invalid = module.weights.clone()
        invalid[0, 0, 0, 0] = 0.0
        invalid[0, 0, 0, 1] = 1.0
        with pytest.raises(ValueError):
            module.weights = invalid

    def test_set_invalid_log_weights_shape_raises(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        with pytest.raises(ValueError):
            module.log_weights = torch.zeros(1, 2, 3)

    def test_feature_to_scope_single_input(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        mapping = module.feature_to_scope
        assert mapping.shape == (2, 1)
        assert set(mapping[0, 0].query) == {0, 1}
        assert set(mapping[1, 0].query) == {2, 3}

    def test_feature_to_scope_two_inputs(self):
        module = make_linsum_two_inputs(2, 3, 2, 1)
        mapping = module.feature_to_scope
        assert mapping.shape == (2, 1)
        assert set(mapping[0, 0].query) == {0, 2}
        assert set(mapping[1, 0].query) == {1, 3}

    def test_split_mode_factory_path_is_used(self):
        leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        module = LinsumLayer(inputs=leaf, out_channels=3, split_mode=SplitMode.interleaved(num_splits=2))
        assert isinstance(module.inputs, SplitInterleaved)

    def test_sample_requires_repetition_index_for_multi_rep(self):
        module = make_linsum_single_input(2, 3, 4, 2)
        num_samples = 5
        data = torch.full((num_samples, 4), torch.nan)
        channel_index = torch.zeros((num_samples, module.out_shape.features), dtype=torch.long)
        mask = torch.ones((num_samples, module.out_shape.features), dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

        with pytest.raises(ValueError):
            module.sample(data=data, sampling_ctx=sampling_ctx)

    def test_sample_two_inputs_uses_cached_log_likelihoods(self):
        module = make_linsum_two_inputs(2, 3, 4, 2)
        batch_size = 8
        data = torch.full((batch_size, 8), torch.nan)
        channel_index = torch.randint(0, module.out_shape.channels, (batch_size, module.out_shape.features))
        mask = torch.ones((batch_size, module.out_shape.features), dtype=torch.bool)
        repetition_index = torch.randint(0, module.out_shape.repetitions, (batch_size,))
        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )

        left_ll = torch.randn(
            batch_size, module.out_shape.features, module.in_shape.channels, module.out_shape.repetitions
        )
        right_ll = torch.randn_like(left_ll)
        cache = Cache()
        cache["log_likelihood"][module.inputs[0]] = left_ll
        cache["log_likelihood"][module.inputs[1]] = right_ll

        samples = module.sample(data=data, cache=cache, sampling_ctx=sampling_ctx, is_mpe=True)
        assert samples.shape == data.shape
        assert torch.isfinite(samples[:, module.scope.query]).all()

    def test_expectation_maximization_creates_cache_if_none(self):
        module = make_linsum_single_input(2, 3, 4, 1)
        data = make_normal_data(out_features=4, num_samples=6)

        with pytest.raises(ValueError):
            module.expectation_maximization(data, cache=None)

    def test_mle_delegates_to_em(self, monkeypatch):
        module = make_linsum_single_input(2, 3, 4, 1)
        data = make_normal_data(out_features=4, num_samples=4)
        cache = Cache()
        called = {}

        def _fake_em(data_arg, bias_correction=True, cache=None):
            called["data"] = data_arg
            called["bias_correction"] = bias_correction
            called["cache"] = cache

        monkeypatch.setattr(module, "expectation_maximization", _fake_em)

        module.maximum_likelihood_estimation(data, bias_correction=False, cache=cache)
        assert called["data"] is data
        assert called["bias_correction"] is False
        assert called["cache"] is cache

    def test_marginalize_two_inputs_branch_outcomes(self, monkeypatch):
        module = make_linsum_two_inputs(2, 3, 4, 1)

        monkeypatch.setattr(module.inputs[0], "marginalize", lambda *args, **kwargs: None)
        monkeypatch.setattr(module.inputs[1], "marginalize", lambda *args, **kwargs: None)
        assert module.marginalize([0]) is None

        module = make_linsum_two_inputs(2, 3, 4, 1)
        right_keep = module.inputs[1]
        monkeypatch.setattr(module.inputs[0], "marginalize", lambda *args, **kwargs: None)
        monkeypatch.setattr(module.inputs[1], "marginalize", lambda *args, **kwargs: right_keep)
        assert module.marginalize([0]) is right_keep

        module = make_linsum_two_inputs(2, 3, 4, 1)
        left_keep = module.inputs[0]
        monkeypatch.setattr(module.inputs[0], "marginalize", lambda *args, **kwargs: left_keep)
        monkeypatch.setattr(module.inputs[1], "marginalize", lambda *args, **kwargs: None)
        assert module.marginalize([0]) is left_keep

        module = make_linsum_two_inputs(2, 3, 4, 1)
        monkeypatch.setattr(module.inputs[0], "marginalize", lambda *args, **kwargs: module.inputs[0])
        monkeypatch.setattr(module.inputs[1], "marginalize", lambda *args, **kwargs: module.inputs[1])
        rebuilt = module.marginalize([0])
        assert isinstance(rebuilt, LinsumLayer)
        assert rebuilt._two_inputs

    def test_marginalize_single_input_branch_outcomes(self, monkeypatch):
        module = make_linsum_single_input(2, 3, 4, 1)

        monkeypatch.setattr(module.inputs.inputs, "marginalize", lambda *args, **kwargs: None)
        assert module.marginalize([0]) is None

        module = make_linsum_single_input(2, 3, 4, 1)
        small_input = make_normal_leaf(out_features=1, out_channels=2, num_repetitions=1)
        monkeypatch.setattr(module.inputs.inputs, "marginalize", lambda *args, **kwargs: small_input)
        assert module.marginalize([0]) is small_input

        module = make_linsum_single_input(2, 3, 6, 1)
        even_input = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
        monkeypatch.setattr(module.inputs.inputs, "marginalize", lambda *args, **kwargs: even_input)
        rebuilt = module.marginalize([0])
        assert isinstance(rebuilt, LinsumLayer)
