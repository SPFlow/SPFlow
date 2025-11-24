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
