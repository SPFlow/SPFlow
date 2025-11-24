import pytest
import torch
from torch import nn

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.base import Module
from spflow.modules.rat.rat_mixing_layer import MixingLayer


class DummyInput(Module):
    """Minimal input module to drive MixingLayer behavior."""

    def __init__(self, out_channels: int = 2, num_repetitions: int = 2, out_features: int = 1):
        super().__init__()
        self._out_channels = out_channels
        self._num_repetitions = num_repetitions
        self._out_features = out_features
        self.scope = Scope(list(range(out_features)))

    @property
    def feature_to_scope(self):
        return [self.scope]

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def num_repetitions(self) -> int:
        return self._num_repetitions

    def log_likelihood(self, data, cache=None):
        batch = data.shape[0]
        # Shape matches expected (B, F, OC, R) orientation used in MixingLayer.
        return torch.zeros(
            batch, self.out_features, self.out_channels, self.num_repetitions, device=data.device
        )

    def sample(self, *args, **kwargs):
        data = kwargs.get("data")
        if data is None:
            data = torch.full((1, len(self.scope.query)), torch.nan, device=self.device)
        data[:, self.scope.query] = 0.0
        return data

    def expectation_maximization(self, data, cache=None):
        return None

    def maximum_likelihood_estimation(self, data, weights=None, cache=None):
        return None

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return self


def test_mixing_layer_initialization_validates():
    inputs = DummyInput(out_channels=2, num_repetitions=2)
    layer = MixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

    assert layer.out_channels == 2
    assert layer.out_features == 1
    assert layer.num_repetitions == 2


def test_mixing_layer_rejects_out_channel_mismatch():
    inputs = DummyInput(out_channels=2, num_repetitions=1)
    try:
        MixingLayer(inputs=inputs, out_channels=3, num_repetitions=1)
    except ValueError as exc:
        assert "out_channels must match" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched out_channels")


def test_mixing_layer_log_likelihood_shape():
    inputs = DummyInput(out_channels=2, num_repetitions=2)
    layer = MixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)
    data = torch.randn(4, 1)

    ll = layer.log_likelihood(data)

    assert ll.shape == (data.shape[0], layer.out_features, layer.out_channels, 1)
    assert torch.isfinite(ll).all()


def test_mixing_layer_sample_uses_cached_posterior():
    inputs = DummyInput(out_channels=1, num_repetitions=1)
    layer = MixingLayer(inputs=inputs, out_channels=1, num_repetitions=1)
    data = torch.full((1, 1), torch.nan)

    cache = {"log_likelihood": {inputs: torch.zeros(1, 1, 1, 1)}}

    sampled = layer.sample(data=data, cache=cache)

    assert sampled.shape == data.shape
    assert torch.isfinite(sampled).all()


def test_mixing_layer_rejects_missing_inputs():
    try:
        MixingLayer(inputs=None, out_channels=1, num_repetitions=1)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "requires at least one input" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing inputs")


def test_mixing_layer_rejects_nonpositive_out_channels():
    inputs = DummyInput(out_channels=1, num_repetitions=1)
    with pytest.raises(ValueError):
        MixingLayer(inputs=inputs, out_channels=0, num_repetitions=1)


def test_mixing_layer_rejects_feature_count_not_one():
    bad_inputs = DummyInput(out_channels=1, num_repetitions=1, out_features=2)
    with pytest.raises(ValueError):
        MixingLayer(inputs=bad_inputs, out_channels=1, num_repetitions=1)


def test_mixing_layer_conflicting_weights_and_out_channels():
    inputs = DummyInput(out_channels=1, num_repetitions=1)
    weights = torch.ones((1, 1, 1))
    with pytest.raises(InvalidParameterCombinationError):
        MixingLayer(inputs=inputs, out_channels=2, num_repetitions=1, weights=weights)
