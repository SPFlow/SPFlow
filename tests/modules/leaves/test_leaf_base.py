import sys
import types

import pytest
import torch
from torch import nn

from spflow.meta import Scope
from spflow.modules.leaves.base import LeafModule


class SimpleParameterNet(nn.Module):
    """Parameter network that returns fixed loc/scale tensors for testing."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls += 1
        loc = torch.full((evidence.shape[0], 1, 1, 1), self.value, device=evidence.device)
        scale = torch.full_like(loc, 0.5)
        return {"loc": loc, "scale": scale}


class TinyLeaf(LeafModule):
    """Compact Normal leaf for exercising base helpers."""

    _torch_distribution_class = torch.distributions.Normal

    def __init__(
        self,
        scope: Scope,
        out_channels: int = 1,
        num_repetitions: int = 1,
        parameter_network: nn.Module | None = None,
        type_error_scale: bool = False,
        override_compute: bool = True,
    ):
        self.type_error_scale = type_error_scale
        self.override_compute = override_compute
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            parameter_network=parameter_network,
        )
        self.loc = nn.Parameter(torch.zeros((len(scope.query), out_channels, num_repetitions)))
        self._scale = nn.Parameter(torch.ones((len(scope.query), out_channels, num_repetitions)))
        self._event_shape = self.loc.shape

    @property
    def _supported_value(self) -> float:
        return 0.0

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @scale.setter
    def scale(self, value: torch.Tensor) -> None:
        if self.type_error_scale:
            raise TypeError("blocked setter for test")
        self._scale.data = value

    def params(self) -> dict[str, torch.Tensor]:
        return {"loc": self.loc, "scale": self.scale}

    def _mle_update_statistics(
        self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool
    ) -> None:
        weighted_mean = (weights * data).sum(0) / weights.sum(0)
        self.loc.data = self._broadcast_to_event_shape(weighted_mean)
        self.scale = self._broadcast_to_event_shape(torch.ones_like(weighted_mean))

    def _compute_parameter_estimates(
        self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool
    ) -> dict[str, torch.Tensor]:
        if not self.override_compute:
            return super()._compute_parameter_estimates(data, weights, bias_correction)
        weighted_mean = (weights * data).sum(0) / weights.sum(0)
        return {"loc": weighted_mean, "scale": torch.ones_like(weighted_mean)}


class NoComputeLeaf(LeafModule):
    """Leaf that relies on base _compute_parameter_estimates implementation."""

    _torch_distribution_class = torch.distributions.Normal

    def __init__(self, scope: Scope):
        super().__init__(scope=scope, out_channels=1, num_repetitions=1)
        self.loc = nn.Parameter(torch.zeros((len(scope.query), 1, 1)))
        self.scale_param = nn.Parameter(torch.ones((len(scope.query), 1, 1)))
        self._event_shape = self.loc.shape

    @property
    def _supported_value(self) -> float:
        return 0.0

    def params(self) -> dict[str, torch.Tensor]:
        return {"loc": self.loc, "scale": self.scale_param}

    def _mle_update_statistics(
        self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool
    ) -> None:
        self.loc.data = self._broadcast_to_event_shape((weights * data).sum(0) / weights.sum(0))

    def _compute_parameter_estimates(
        self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("_compute_parameter_estimates not implemented for testing")


class TrackingLeaf(TinyLeaf):
    """TinyLeaf variant that records compute calls for KMeans paths."""

    def __init__(self, scope: Scope, out_channels: int = 2):
        super().__init__(scope=scope, out_channels=out_channels, override_compute=True)
        self.compute_calls: list[int] = []

    def _compute_parameter_estimates(
        self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool
    ) -> dict[str, torch.Tensor]:
        self.compute_calls.append(data.shape[0])
        return super()._compute_parameter_estimates(data, weights, bias_correction)


def test_conditional_distribution_uses_parameter_network():
    """Verify conditional_distribution delegates parameter creation to the network."""
    scope = Scope([0])
    param_net = SimpleParameterNet(value=2.5)
    leaf = TinyLeaf(scope=scope, parameter_network=param_net)
    evidence = torch.ones((4, 1))

    dist = leaf.conditional_distribution(evidence)

    assert param_net.calls == 1
    torch.testing.assert_close(dist.loc, torch.full((4, 1, 1, 1), 2.5))
    torch.testing.assert_close(dist.scale, torch.full((4, 1, 1, 1), 0.5))
    with pytest.raises(ValueError):
        leaf.conditional_distribution(evidence=None)


def test_compute_parameter_estimates_not_implemented():
    """Base class raises when _compute_parameter_estimates is not overridden."""
    scope = Scope([0])
    leaf = NoComputeLeaf(scope=scope)
    data = torch.randn(3, 1)
    weights = torch.ones((3, 1))

    with pytest.raises(NotImplementedError):
        leaf._compute_parameter_estimates(data, weights, bias_correction=False)


def test_set_mle_parameters_handles_type_error():
    """_set_mle_parameters should fall back to .data assignment on setter errors."""
    scope = Scope([0])
    leaf = TinyLeaf(scope=scope, type_error_scale=True)
    new_loc = torch.full_like(leaf.loc, 3.0)
    new_scale = torch.full_like(leaf.scale, 0.7)

    leaf._set_mle_parameters({"loc": new_loc, "scale": new_scale})

    torch.testing.assert_close(leaf.loc, new_loc)
    torch.testing.assert_close(leaf.scale, new_scale)


def test_mode_property_matches_distribution_mode():
    """Mode property should mirror the underlying distribution mode."""
    scope = Scope([0])
    leaf = TinyLeaf(scope=scope)
    leaf.loc.data = torch.ones_like(leaf.loc) * 1.75

    torch.testing.assert_close(leaf.mode, torch.full_like(leaf.loc, 1.75))


def test_update_parameters_with_kmeans(monkeypatch):
    """KMeans path should assign per-cluster parameter estimates."""
    scope = Scope([0])
    leaf = TinyLeaf(scope=scope, out_channels=2)
    cluster_one = torch.randn(6, 1) * 0.1
    cluster_two = torch.randn(6, 1) * 0.1 + 5.0
    data = torch.cat([cluster_one, cluster_two], dim=0)

    class StubKMeans:
        def __init__(self, n_clusters: int, mode: str, init_method: str):
            self.n_clusters = n_clusters

        def fit_predict(self, inputs: torch.Tensor) -> torch.Tensor:
            midpoint = inputs.mean()
            return (inputs.view(-1) > midpoint).long()

    monkeypatch.setitem(
        sys.modules,
        "fast_pytorch_kmeans",
        types.SimpleNamespace(KMeans=StubKMeans),
    )

    leaf.maximum_likelihood_estimation(data=data, use_kmeans=True)

    assert leaf.loc.shape == (1, 2, 1)
    assert leaf.loc[:, 0, 0] < leaf.loc[:, 1, 0]


def test_update_parameters_with_kmeans_empty_cluster(monkeypatch):
    """Empty clusters should fall back to global estimates."""
    scope = Scope([0])
    leaf = TrackingLeaf(scope=scope, out_channels=2)
    data = torch.randn(5, 1)

    class StubKMeans:
        def __init__(self, n_clusters: int, mode: str, init_method: str):
            self.n_clusters = n_clusters

        def fit_predict(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(inputs.shape[0], dtype=torch.long)

    monkeypatch.setitem(
        sys.modules,
        "fast_pytorch_kmeans",
        types.SimpleNamespace(KMeans=StubKMeans),
    )

    leaf.maximum_likelihood_estimation(data=data, use_kmeans=True)

    assert leaf.compute_calls == [data.shape[0], data.shape[0]]
    torch.testing.assert_close(leaf.loc[:, 0, 0], leaf.loc[:, 1, 0])


def test_mle_rejects_conditional_leaf():
    """Conditional leaves should refuse MLE updates."""
    scope = Scope([0])
    leaf = TinyLeaf(scope=scope, parameter_network=SimpleParameterNet(value=1.0))
    data = torch.randn(4, 1)

    with pytest.raises(RuntimeError):
        leaf.maximum_likelihood_estimation(data=data)


def test_prepare_mle_data_ignore_nan():
    """NaN handling should respect the 'ignore' nan_strategy."""
    scope = Scope([0])
    leaf = TinyLeaf(scope=scope)
    data = torch.tensor([[0.1], [float("nan")], [0.3]])

    scoped, weights = leaf._prepare_mle_data(data, nan_strategy="ignore")

    assert scoped.shape[0] == 2
    assert weights.shape[0] == 2
    assert torch.all(weights.squeeze(-1) > 0)


def test_sample_requires_repetition_index_for_multiple_repetitions():
    """Sampling multi-repetition leaves requires repetition index in context."""
    leaf = TinyLeaf(scope=Scope([0]), num_repetitions=2)

    with pytest.raises(ValueError):
        leaf.sample(num_samples=2)


def test_feature_to_scope():
    """Test feature_to_scope property returns correct array of Scope objects."""
    # Test with 3 features and 2 repetitions
    scope = Scope([0, 1, 2])
    leaf = TinyLeaf(scope=scope, out_channels=3, num_repetitions=2)

    feature_scopes = leaf.feature_to_scope

    # Validate shape equals (3, 2)
    assert feature_scopes.shape == (3, 2), f"Expected shape (3, 2), got {feature_scopes.shape}"

    # Validate all elements are Scope objects
    for i in range(3):
        for j in range(2):
            assert isinstance(
                feature_scopes[i, j], Scope
            ), f"Element at ({i}, {j}) is not a Scope object, got {type(feature_scopes[i, j])}"

    # Validate each feature maps to correct single-element scope
    # Feature 0 should map to Scope([0])
    assert feature_scopes[0, 0] == Scope([0]), f"Feature 0 should map to Scope([0]), got {feature_scopes[0, 0]}"
    # Feature 1 should map to Scope([1])
    assert feature_scopes[1, 0] == Scope([1]), f"Feature 1 should map to Scope([1]), got {feature_scopes[1, 0]}"
    # Feature 2 should map to Scope([2])
    assert feature_scopes[2, 0] == Scope([2]), f"Feature 2 should map to Scope([2]), got {feature_scopes[2, 0]}"

    # Validate all repetitions have identical scope mappings
    for i in range(3):
        for j in range(1, 2):
            assert (
                feature_scopes[i, 0] == feature_scopes[i, j]
            ), f"Feature {i} has different scopes across repetitions: {feature_scopes[i, 0]} vs {feature_scopes[i, j]}"


def test_feature_to_scope_single_repetition():
    """Test feature_to_scope with single repetition (common case)."""
    scope = Scope([5, 10, 15])
    leaf = TinyLeaf(scope=scope, out_channels=1, num_repetitions=1)

    feature_scopes = leaf.feature_to_scope

    # Shape should be (3, 1)
    assert feature_scopes.shape == (3, 1)

    # Validate mapping for non-contiguous scope
    assert feature_scopes[0, 0] == Scope([5])
    assert feature_scopes[1, 0] == Scope([10])
    assert feature_scopes[2, 0] == Scope([15])


def test_feature_to_scope_single_feature():
    """Test feature_to_scope with single feature."""
    scope = Scope([42])
    leaf = TinyLeaf(scope=scope, out_channels=2, num_repetitions=3)

    feature_scopes = leaf.feature_to_scope

    # Shape should be (1, 3)
    assert feature_scopes.shape == (1, 3)

    # All repetitions should map to Scope([42])
    for j in range(3):
        assert feature_scopes[0, j] == Scope([42])


def test_feature_to_scope_many_repetitions():
    """Test feature_to_scope with many repetitions."""
    scope = Scope([0, 1])
    num_repetitions = 5
    leaf = TinyLeaf(scope=scope, out_channels=2, num_repetitions=num_repetitions)

    feature_scopes = leaf.feature_to_scope

    # Shape should be (2, 5)
    assert feature_scopes.shape == (2, num_repetitions)

    # Verify consistency across all repetitions
    for rep_idx in range(num_repetitions):
        assert feature_scopes[0, rep_idx] == Scope([0])
        assert feature_scopes[1, rep_idx] == Scope([1])


def test_feature_to_scope_array_type():
    """Test that feature_to_scope returns numpy array with proper dtype."""
    import numpy as np

    scope = Scope([0, 1, 2])
    leaf = TinyLeaf(scope=scope, out_channels=1, num_repetitions=2)

    feature_scopes = leaf.feature_to_scope

    # Verify it's a numpy array
    assert isinstance(feature_scopes, np.ndarray), f"Expected np.ndarray, got {type(feature_scopes)}"

    # Verify dtype is object (since it contains Scope objects)
    assert feature_scopes.dtype == object or feature_scopes.dtype == Scope, (
        f"Expected dtype object or Scope, got {feature_scopes.dtype}"
    )
