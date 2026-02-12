"""Tests for WeightedSum module."""

import numpy as np
import pytest
import torch

from spflow.exceptions import ShapeError
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.sampling_context import SamplingContext
from spflow.modules.leaves.normal import Normal
from spflow.zoo.pic.weighted_sum import WeightedSum


class DummyInput(Module):
    """Test helper module with deterministic shape/scope behavior."""

    def __init__(
        self,
        feature_to_scope: np.ndarray,
        channels: int,
        repetitions: int,
        marginalize_result: Module | None = None,
    ) -> None:
        super().__init__()
        self._feature_to_scope = feature_to_scope
        self.in_shape = ModuleShape(
            features=feature_to_scope.shape[0], channels=channels, repetitions=repetitions
        )
        self.out_shape = self.in_shape
        query_rvs: set[int] = set()
        for entry in feature_to_scope.flatten():
            if entry is not None:
                query_rvs.update(entry.query)
        self.scope = Scope(sorted(query_rvs))
        self._marginalize_result = marginalize_result
        self.sample_calls: list[SamplingContext] = []

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self._feature_to_scope

    def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:
        batch_size = data.shape[0]
        return torch.zeros(
            batch_size,
            self.out_shape.features,
            self.out_shape.channels,
            self.out_shape.repetitions,
            dtype=data.dtype,
            device=data.device,
        )

    def sample(
        self,
        num_samples: int | None = None,
        data: torch.Tensor | None = None,
        is_mpe: bool = False,
        cache=None,
        sampling_ctx: SamplingContext | None = None,
    ) -> torch.Tensor:
        self.sample_calls.append(sampling_ctx)
        return data

    def _sample(
        self,
        data: torch.Tensor,
        sampling_ctx: SamplingContext,
        cache,
        is_mpe: bool = False,
    ) -> torch.Tensor:
        del cache
        del is_mpe
        self.sample_calls.append(sampling_ctx)
        return data

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return self._marginalize_result


class TestWeightedSumInit:
    """Tests for WeightedSum initialization."""

    def test_basic_init(self):
        """Test basic initialization with a single input."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(1, 3, 2, 1)  # (F, IC, OC, R)

        ws = WeightedSum(inputs=leaf, weights=weights)

        assert ws.out_shape.channels == 2
        assert ws.out_shape.features == 1
        assert ws.out_shape.repetitions == 1

    def test_init_1d_weights(self):
        """Test initialization with 1D weights (broadcasts to 4D)."""
        leaf = Normal(scope=Scope([0]), out_channels=4)
        weights = torch.ones(4)  # Will become (1, 4, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        assert ws._weights.shape == (1, 4, 1, 1)

    def test_init_2d_weights(self):
        """Test initialization with 2D weights."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(3, 2)  # Will become (1, 3, 2, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        assert ws._weights.shape == (1, 3, 2, 1)

    def test_init_multiple_inputs(self):
        """Test initialization with multiple inputs (will be concatenated)."""
        leaf1 = Normal(scope=Scope([0]), out_channels=2)
        leaf2 = Normal(scope=Scope([0]), out_channels=2)
        weights = torch.ones(1, 4, 1, 1)  # 4 = 2 + 2 (concatenated)

        ws = WeightedSum(inputs=[leaf1, leaf2], weights=weights)

        assert ws.in_shape.channels == 4  # Concatenated

    def test_init_empty_inputs_raises(self):
        """Test that empty inputs raises ValueError."""
        with pytest.raises(ValueError):
            WeightedSum(inputs=[], weights=torch.ones(1))

    def test_init_invalid_weight_dim_raises(self):
        """Test that 5D+ weights raises ShapeError."""
        from spflow.exceptions import ShapeError

        leaf = Normal(scope=Scope([0]), out_channels=2)
        weights = torch.ones(1, 1, 1, 1, 1)  # 5D

        with pytest.raises(ShapeError):
            WeightedSum(inputs=leaf, weights=weights)

    def test_init_list_weights_and_single_input_list(self):
        """Test list weights conversion and single-list input unwrapping."""
        leaf = Normal(scope=Scope([0]), out_channels=2)
        ws = WeightedSum(inputs=[leaf], weights=[1.0, 2.0])
        assert ws.inputs is leaf
        assert ws.weights.shape == (1, 2, 1, 1)

    def test_init_3d_weights_and_feature_broadcast(self):
        """Test 3D weight expansion plus feature broadcasting."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        weights = torch.ones(1, 2, 3)  # (1, IC, OC) -> unsqueeze + repeat over 2 features
        ws = WeightedSum(inputs=leaf, weights=weights)
        assert ws.weights.shape == (2, 2, 3, 1)

    def test_init_negative_weights_raises(self):
        """Test non-negative constraint for weights."""
        from spflow.exceptions import InvalidWeightsError

        leaf = Normal(scope=Scope([0]), out_channels=2)
        with pytest.raises(InvalidWeightsError):
            WeightedSum(inputs=leaf, weights=torch.tensor([-1.0, 1.0]))

    def test_init_feature_mismatch_raises(self):
        """Test feature mismatch shape validation."""
        from spflow.exceptions import ShapeError

        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        weights = torch.ones(3, 2, 1, 1)
        with pytest.raises(ShapeError):
            WeightedSum(inputs=leaf, weights=weights)

    def test_init_in_channel_mismatch_raises(self):
        """Test in-channel mismatch shape validation."""
        from spflow.exceptions import ShapeError

        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(1, 2, 1, 1)
        with pytest.raises(ShapeError):
            WeightedSum(inputs=leaf, weights=weights)


class TestWeightedSumNoNormalization:
    """Tests verifying weights are NOT normalized."""

    def test_weights_preserved_exactly(self):
        """Test that weights are stored exactly as provided."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.tensor([0.1, 0.2, 0.9]).view(1, 3, 1, 1)  # (F, IC, OC, R)

        ws = WeightedSum(inputs=leaf, weights=weights)

        # Weights should NOT be normalized to sum to 1 per channel
        assert torch.allclose(ws.weights, weights)
        # These weights don't sum to 1 and that should be fine
        assert not torch.allclose(ws.weights.sum(dim=1), torch.ones(1, 1, 1))

    def test_unnormalized_weights_allowed(self):
        """Test that quadrature-style weights (not summing to 1) work."""
        leaf = Normal(scope=Scope([0]), out_channels=5)
        # Quadrature weights example (e.g., Gauss-Legendre)
        quadrature_weights = torch.tensor([0.2369, 0.4786, 0.5688, 0.4786, 0.2369])
        ws_weights = quadrature_weights.view(1, 5, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=ws_weights)

        # These sum to ~2.0, not 1.0, and should be preserved
        assert torch.allclose(ws.weights.sum(), torch.tensor(2.0), atol=0.01)

    def test_log_weights_is_log_of_raw(self):
        """Test that log_weights returns log of raw weights, not log-softmax."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        expected_log_weights = torch.log(weights)
        assert torch.allclose(ws.log_weights, expected_log_weights)

    def test_log_weights_allows_structural_zeros(self):
        """Test that structural zeros produce -inf log-weights (needed for sparse mixing matrices)."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.tensor([1.0, 0.0, 2.0]).view(1, 3, 1, 1)  # (F, IC, OC, R)

        ws = WeightedSum(inputs=leaf, weights=weights)

        assert torch.isneginf(ws.log_weights).any()


class TestWeightedSumLogLikelihood:
    """Tests for log-likelihood computation."""

    def test_log_likelihood_shape(self):
        """Test log-likelihood output shape."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=3)
        weights = torch.ones(2, 3, 2, 1)  # 2 features, 3 in_channels, 2 out_channels

        ws = WeightedSum(inputs=leaf, weights=weights)

        data = torch.randn(10, 2)  # batch=10, features=2
        ll = ws.log_likelihood(data)

        assert ll.shape == (10, 2, 2, 1)  # (batch, features, out_channels, repetitions)

    def test_log_likelihood_uses_raw_weights(self):
        """Test that log-likelihood uses unnormalized weights."""
        # Create simple setup
        leaf = Normal(scope=Scope([0]), out_channels=1)
        # Weight of 2.0 (not normalized)
        weights = torch.tensor([[[[2.0]]]])

        ws = WeightedSum(inputs=leaf, weights=weights)

        data = torch.tensor([[0.0]])
        ll = ws.log_likelihood(data)

        # log(2 * p(x)) = log(2) + log(p(x))
        leaf_ll = leaf.log_likelihood(data)
        expected = torch.logsumexp(leaf_ll.unsqueeze(3) + torch.log(weights), dim=2)

        assert torch.allclose(ll, expected, atol=1e-5)


class TestWeightedSumSetWeights:
    """Tests for setting weights."""

    def test_set_weights(self):
        """Test setting weights directly."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(1, 3, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        new_weights = torch.full((1, 3, 1, 1), 0.5)
        ws.weights = new_weights

        assert torch.allclose(ws.weights, new_weights)

    def test_set_weights_wrong_shape_raises(self):
        """Test that setting weights with wrong shape raises error."""
        from spflow.exceptions import ShapeError

        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(1, 3, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        with pytest.raises(ShapeError):
            ws.weights = torch.ones(1, 2, 1, 1)  # Wrong shape


class TestWeightedSumSamplingAndMarginalize:
    """Tests for sampling and marginalization code paths."""

    def test_feature_to_scope_and_extra_repr(self):
        """Test delegated feature_to_scope and string repr."""
        f2s = np.array([[Scope([0])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=1)
        ws = WeightedSum(inputs=inp, weights=torch.ones(1, 1, 1, 1))
        assert ws.feature_to_scope.shape == (1, 1)
        assert "weights=(1, 1, 1, 1)" in ws.extra_repr()

    def test_log_likelihood_wrapped_function_cache_none_branch(self):
        """Call undecorated method to exercise internal cache initialization."""
        f2s = np.array([[Scope([0])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=1)
        ws = WeightedSum(inputs=inp, weights=torch.ones(1, 1, 1, 1))
        data = torch.zeros(2, 1)
        ll = WeightedSum.log_likelihood.__wrapped__(ws, data, cache=None)
        assert ll.shape == (2, 1, 1, 1)

    def test_sample_with_default_context_and_mpe(self):
        """Test sample path with default context and MPE argmax selection."""
        f2s = np.array([[Scope([0]), Scope([0])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=2, repetitions=2)
        weights = torch.tensor([[[[1.0, 3.0], [5.0, 2.0]], [[2.0, 1.0], [4.0, 7.0]]]])
        ws = WeightedSum(inputs=inp, weights=weights)

        sampling_ctx = SamplingContext(
            channel_index=torch.zeros((3, 1), dtype=torch.long),
            mask=torch.ones((3, 1), dtype=torch.bool),
            repetition_index=torch.tensor([1, 0, 1], dtype=torch.long),
        )
        out = ws.sample(data=torch.full((3, 1), float("nan")), is_mpe=True, sampling_ctx=sampling_ctx)

        assert out.shape == (3, 1)
        assert inp.sample_calls[-1].channel_index.shape == (3, 1)

    def test_sample_requires_repetition_index_when_repetitions_gt_1(self):
        """Test validation when repetitions > 1 and repetition_idx is missing."""
        f2s = np.array([[Scope([0]), Scope([0])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=2, repetitions=2)
        ws = WeightedSum(inputs=inp, weights=torch.ones(1, 2, 2, 2))
        sampling_ctx = SamplingContext(
            channel_index=torch.zeros((2, 1), dtype=torch.long),
            mask=torch.ones((2, 1), dtype=torch.bool),
        )
        with pytest.raises(ValueError):
            ws.sample(data=torch.full((2, 1), float("nan")), sampling_ctx=sampling_ctx)

    def test_sample_raises_for_zero_rows(self):
        """Test stochastic sample branch rejects zero-sum weight rows."""
        f2s = np.array([[Scope([0])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=2, repetitions=1)
        ws = WeightedSum(inputs=inp, weights=torch.zeros(1, 2, 2, 1))
        sampling_ctx = SamplingContext(
            channel_index=torch.zeros((2, 1), dtype=torch.long),
            mask=torch.ones((2, 1), dtype=torch.bool),
        )
        with pytest.raises(ShapeError, match="zero-sum routing weights"):
            ws.sample(data=torch.full((2, 1), float("nan")), sampling_ctx=sampling_ctx)

    def test_sample_creates_data_when_none(self):
        """Test num_samples/data default creation branch."""
        f2s = np.array([[Scope([0])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=1)
        ws = WeightedSum(inputs=inp, weights=torch.ones(1, 1, 1, 1))
        out = ws.sample()
        assert out.shape == (1, 1)

    def test_sample_raises_on_incompatible_mask_width(self):
        """Test strict mask-width validation when routing shape changes."""
        f2s = np.array([[Scope([0]), Scope([0])], [Scope([1]), Scope([1])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=2)
        ws = WeightedSum(inputs=inp, weights=torch.ones(2, 1, 3, 2))
        sampling_ctx = SamplingContext(
            channel_index=torch.zeros((2, 2), dtype=torch.long),
            mask=torch.ones((2, 2), dtype=torch.bool),
            repetition_index=torch.zeros(2, dtype=torch.long),
        )
        # Deliberately introduce a narrower mask (bypassing public setter invariants)
        # so WeightedSum hits strict update-time shape validation.
        sampling_ctx._mask = torch.ones((2, 1), dtype=torch.bool)  # type: ignore[attr-defined]
        with pytest.raises(ShapeError, match="incompatible feature width"):
            ws.sample(data=torch.full((2, 2), float("nan")), is_mpe=True, sampling_ctx=sampling_ctx)

    def test_marginalize_full_scope_returns_none(self):
        """Test full marginalization short-circuit."""
        f2s = np.array([[Scope([0])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=1, marginalize_result=None)
        ws = WeightedSum(inputs=inp, weights=torch.ones(1, 1, 1, 1))
        assert ws.marginalize([0]) is None

    def test_marginalize_no_overlap_keeps_input(self):
        """Test no-overlap marginalization path."""
        f2s = np.array([[Scope([0])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=1, marginalize_result=None)
        ws = WeightedSum(inputs=inp, weights=torch.ones(1, 1, 1, 1))
        marg = ws.marginalize([2])
        assert isinstance(marg, WeightedSum)
        assert marg.inputs is inp

    def test_marginalize_partial_with_consistent_masks(self):
        """Test partial marginalization path with equal masked feature counts."""
        f2s = np.array([[Scope([0, 1]), Scope([0, 1])], [Scope([0, 2]), Scope([0, 2])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=2)
        inp._marginalize_result = inp
        ws = WeightedSum(inputs=inp, weights=torch.ones(2, 1, 1, 2))
        marg = ws.marginalize([0])
        assert isinstance(marg, WeightedSum)
        assert marg.weights.shape == (2, 1, 1, 2)

    def test_marginalize_partial_with_padding(self):
        """Test partial marginalization path that pads unequal feature counts."""
        f2s = np.array([[Scope([0]), Scope([1])], [Scope([0, 1]), Scope([1])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=2)
        inp._marginalize_result = inp
        ws = WeightedSum(inputs=inp, weights=torch.ones(2, 1, 1, 2))
        marg = ws.marginalize([0])
        assert isinstance(marg, WeightedSum)
        assert marg.weights.shape == (2, 1, 1, 2)

    def test_marginalize_partial_when_child_returns_none(self):
        """Test partial marginalization returning None if child is removed."""
        f2s = np.array([[Scope([0, 1]), Scope([0, 1])]], dtype=object)
        inp = DummyInput(feature_to_scope=f2s, channels=1, repetitions=2, marginalize_result=None)
        ws = WeightedSum(inputs=inp, weights=torch.ones(1, 1, 1, 2))
        assert ws.marginalize([0]) is None
