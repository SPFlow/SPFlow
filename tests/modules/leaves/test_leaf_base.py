import pytest
import torch
from torch import nn

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.meta import Scope
from spflow.meta.data.interval_evidence import IntervalEvidence
from spflow.modules.leaves import Categorical
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext


class SimpleParameterNet(nn.Module):
    """Parameter network that returns fixed loc/scale tensors for testing."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls += 1
        loc = torch.full((evidence.shape[0], 1, 1, 1), self.value)
        scale = torch.full_like(loc, 0.5)
        return {"loc": loc, "scale": scale}


class TinyLeaf(LeafModule):
    """Compact Normal leaf for exercising base helpers."""

    _torch_distribution_class = torch.distributions.Normal

    @property
    def _torch_distribution_class_with_differentiable_sampling(self):
        return torch.distributions.Normal

    def __init__(
        self,
        scope: Scope,
        out_channels: int = 1,
        num_repetitions: int = 1,
        parameter_fn: nn.Module | None = None,
        type_error_scale: bool = False,
        override_compute: bool = True,
    ):
        self.type_error_scale = type_error_scale
        self.override_compute = override_compute
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            parameter_fn=parameter_fn,
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


class SupportedValueLeaf(TinyLeaf):
    """TinyLeaf variant with configurable supported value for imputation tests."""

    def __init__(self, scope: Scope, supported_value):
        self._supported_value_value = supported_value
        super().__init__(scope=scope)

    @property
    def _supported_value(self):
        return self._supported_value_value


class NoCdfDistribution:
    """Distribution stub that intentionally lacks cdf()."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args=None):
        self.loc = loc
        self.scale = scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value)


class RaisingCdfDistribution(NoCdfDistribution):
    """Distribution stub with cdf() that raises NotImplementedError."""

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("cdf not implemented")


class NoCdfLeaf(TinyLeaf):
    _torch_distribution_class = NoCdfDistribution


class RaisingCdfLeaf(TinyLeaf):
    _torch_distribution_class = RaisingCdfDistribution


def test_conditional_distribution_uses_parameter_fn():
    """Verify conditional_distribution delegates parameter creation to the network."""
    scope = Scope([0])
    param_net = SimpleParameterNet(value=2.5)
    leaf = TinyLeaf(scope=scope, parameter_fn=param_net)
    evidence = torch.ones((4, 1))

    dist = leaf.conditional_distribution(evidence)

    assert param_net.calls == 1
    torch.testing.assert_close(dist.loc, torch.full((4, 1, 1, 1), 2.5))
    torch.testing.assert_close(dist.scale, torch.full((4, 1, 1, 1), 0.5))
    with pytest.raises(ValueError):
        leaf.conditional_distribution(evidence=None)


def test_mode_property_matches_distribution_mode():
    """Mode property should mirror the underlying distribution mode."""
    scope = Scope([0])
    leaf = TinyLeaf(scope=scope)
    leaf.loc.data = torch.ones_like(leaf.loc) * 1.75

    torch.testing.assert_close(leaf.mode(), torch.full_like(leaf.loc, 1.75))


def test_distribution_ignores_differentiable_flag_by_default():
    """Base distribution implementation should keep default behavior unchanged."""
    leaf = TinyLeaf(scope=Scope([0]), out_channels=2, num_repetitions=3)

    dist_default = leaf.distribution()
    dist_differentiable = leaf.distribution(with_differentiable_sampling=True)

    torch.testing.assert_close(dist_default.loc, dist_differentiable.loc)
    torch.testing.assert_close(dist_default.scale, dist_differentiable.scale)


def test_distribution_raises_without_differentiable_override():
    leaf = NoComputeLeaf(scope=Scope([0]))

    with pytest.raises(NotImplementedError):
        leaf.distribution(with_differentiable_sampling=True)


def test_sample_forwards_is_differentiable_to_distribution(monkeypatch):
    class DistWithRsample:
        has_rsample = True

        def sample(self, sample_shape):
            raise AssertionError("Expected rsample() in differentiable sampling mode.")

        def rsample(self, sample_shape):
            n = int(sample_shape[0])
            return torch.full((n, 1, 1, 1), 2.0)

    leaf = TinyLeaf(scope=Scope([0]))
    seen_flags: list[bool] = []

    def fake_distribution(with_differentiable_sampling: bool = False):
        seen_flags.append(with_differentiable_sampling)
        return DistWithRsample()

    monkeypatch.setattr(leaf, "distribution", fake_distribution)

    data = torch.tensor([[float("nan")], [float("nan")]])
    sampling_ctx = SamplingContext(
        channel_index=torch.ones((2, 1, 1), dtype=torch.float32),
        mask=torch.ones((2, 1), dtype=torch.bool),
        repetition_index=torch.ones((2, 1), dtype=torch.float32),
        is_mpe=False,
        is_differentiable=True,
    )
    out = leaf._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert seen_flags == [True]
    assert out.shape == data.shape


def test_sample_forwards_is_differentiable_to_conditional_distribution(monkeypatch):
    class DistWithMode:
        def __init__(self, batch_size: int):
            self.mode = torch.full((batch_size, 1), 5.0)

    leaf = TinyLeaf(scope=Scope(query=[0], evidence=[1]), parameter_fn=SimpleParameterNet(value=0.0))
    seen_flags: list[bool] = []

    def fake_conditional_distribution(evidence, with_differentiable_sampling: bool = False):
        seen_flags.append(with_differentiable_sampling)
        return DistWithMode(batch_size=evidence.shape[0])

    monkeypatch.setattr(leaf, "conditional_distribution", fake_conditional_distribution)

    data = torch.tensor([[float("nan"), 1.0], [float("nan"), 2.0]])
    sampling_ctx = SamplingContext(
        channel_index=torch.ones((2, 1, 1), dtype=torch.float32),
        mask=torch.ones((2, 1), dtype=torch.bool),
        repetition_index=torch.ones((2, 1), dtype=torch.float32),
        is_mpe=True,
        is_differentiable=True,
    )
    out = leaf._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert seen_flags == [True]
    assert out.shape == data.shape


def test_mle_rejects_conditional_leaf():
    """Conditional leaves should refuse MLE updates."""
    scope = Scope([0])
    leaf = TinyLeaf(scope=scope, parameter_fn=SimpleParameterNet(value=1.0))
    data = torch.randn(4, 1)

    with pytest.raises(RuntimeError):
        leaf.maximum_likelihood_estimation(data=data)


@pytest.mark.parametrize("is_mpe", [False, True])
def test_sample_accepts_column_vector_repetition_index(is_mpe: bool):
    """Sampling accepts repetition_index with shape (batch, 1)."""
    leaf = TinyLeaf(scope=Scope([0]), out_channels=2, num_repetitions=2)
    data = torch.full((4, 1), float("nan"))
    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((4, 1), dtype=torch.long),
        mask=torch.ones((4, 1), dtype=torch.bool),
        repetition_index=torch.randint(low=0, high=2, size=(4, 1)),
        is_mpe=is_mpe,
    )
    samples = leaf._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert samples.shape == (4, 1)
    assert torch.isfinite(samples).all()


def test_feature_to_scope():
    """Test feature_to_scope property returns correct array of Scope objects."""
    scope = Scope([0, 1, 2])
    leaf = TinyLeaf(scope=scope, out_channels=3, num_repetitions=2)

    feature_scopes = leaf.feature_to_scope

    assert feature_scopes.shape == (3, 2), f"Expected shape (3, 2), got {feature_scopes.shape}"

    for i in range(3):
        for j in range(2):
            assert isinstance(
                feature_scopes[i, j], Scope
            ), f"Element at ({i}, {j}) is not a Scope object, got {type(feature_scopes[i, j])}"

    assert feature_scopes[0, 0] == Scope(
        [0]
    ), f"Feature 0 should map to Scope([0]), got {feature_scopes[0, 0]}"
    assert feature_scopes[1, 0] == Scope(
        [1]
    ), f"Feature 1 should map to Scope([1]), got {feature_scopes[1, 0]}"
    assert feature_scopes[2, 0] == Scope(
        [2]
    ), f"Feature 2 should map to Scope([2]), got {feature_scopes[2, 0]}"

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

    assert feature_scopes.shape == (3, 1)

    assert feature_scopes[0, 0] == Scope([5])
    assert feature_scopes[1, 0] == Scope([10])
    assert feature_scopes[2, 0] == Scope([15])


def test_feature_to_scope_single_feature():
    """Test feature_to_scope with single feature."""
    scope = Scope([42])
    leaf = TinyLeaf(scope=scope, out_channels=2, num_repetitions=3)

    feature_scopes = leaf.feature_to_scope

    assert feature_scopes.shape == (1, 3)

    for j in range(3):
        assert feature_scopes[0, j] == Scope([42])


def test_feature_to_scope_many_repetitions():
    """Test feature_to_scope with many repetitions."""
    scope = Scope([0, 1])
    num_repetitions = 5
    leaf = TinyLeaf(scope=scope, out_channels=2, num_repetitions=num_repetitions)

    feature_scopes = leaf.feature_to_scope

    assert feature_scopes.shape == (2, num_repetitions)

    for rep_idx in range(num_repetitions):
        assert feature_scopes[0, rep_idx] == Scope([0])
        assert feature_scopes[1, rep_idx] == Scope([1])


def test_feature_to_scope_array_type():
    """Test that feature_to_scope returns numpy array with proper dtype."""
    import numpy as np

    scope = Scope([0, 1, 2])
    leaf = TinyLeaf(scope=scope, out_channels=1, num_repetitions=2)

    feature_scopes = leaf.feature_to_scope

    assert isinstance(feature_scopes, np.ndarray), f"Expected np.ndarray, got {type(feature_scopes)}"

    assert (
        feature_scopes.dtype == object or feature_scopes.dtype == Scope
    ), f"Expected dtype object or Scope, got {feature_scopes.dtype}"


def test_inputs_setter_raises_attribute_error():
    leaf = TinyLeaf(scope=Scope([0]))
    with pytest.raises(AttributeError):
        leaf.inputs = []


@pytest.mark.parametrize(
    "supported_value, expected",
    [
        (torch.tensor(3.0), torch.tensor([[3.0, 3.0], [3.0, 3.0]])),
        (torch.tensor([1.0, 2.0]), torch.tensor([[1.0, 2.0], [1.0, 2.0]])),
    ],
)
def test_supported_value_for_imputation_tensor_success(supported_value, expected):
    leaf = SupportedValueLeaf(scope=Scope([0, 1]), supported_value=supported_value)
    scoped_data = torch.zeros((2, 2))
    torch.testing.assert_close(leaf._supported_value_for_imputation(scoped_data), expected)


def test_supported_value_for_imputation_raises_for_invalid_shapes_and_types():
    scoped_data = torch.zeros((2, 2))

    leaf_bad_dim = SupportedValueLeaf(scope=Scope([0, 1]), supported_value=0.0)
    with pytest.raises(ShapeError):
        leaf_bad_dim._supported_value_for_imputation(torch.zeros((2, 2, 1)))

    leaf_bad_type = SupportedValueLeaf(scope=Scope([0, 1]), supported_value={"not": "supported"})
    with pytest.raises(TypeError):
        leaf_bad_type._supported_value_for_imputation(scoped_data)

    leaf_bad_1d = SupportedValueLeaf(scope=Scope([0, 1]), supported_value=torch.tensor([1.0, 2.0, 3.0]))
    with pytest.raises(ShapeError):
        leaf_bad_1d._supported_value_for_imputation(scoped_data)

    leaf_bad_2d = SupportedValueLeaf(scope=Scope([0, 1]), supported_value=torch.ones((1, 2)))
    with pytest.raises(ShapeError):
        leaf_bad_2d._supported_value_for_imputation(scoped_data)


def test_set_mle_parameters_falls_back_to_data_assignment_after_type_error():
    leaf = TinyLeaf(scope=Scope([0]), type_error_scale=True)
    target_scale = torch.full_like(leaf.scale, 2.25)

    leaf._set_mle_parameters({"scale": target_scale})

    torch.testing.assert_close(leaf.scale, target_scale)


def test_event_shape_raises_if_not_initialized():
    leaf = TinyLeaf(scope=Scope([0]))
    leaf._event_shape = None
    with pytest.raises(RuntimeError):
        _ = leaf.event_shape


def test_broadcast_to_event_shape_handles_2d_and_3d_event_shapes():
    two_dim_leaf = TinyLeaf(scope=Scope([0, 1]), out_channels=3, num_repetitions=1)
    two_dim_leaf._event_shape = (2, 3)
    param_est_2d = torch.tensor([1.0, 2.0])
    assert two_dim_leaf._broadcast_to_event_shape(param_est_2d).shape == (2, 3)

    three_dim_leaf = TinyLeaf(scope=Scope([0, 1]), out_channels=2, num_repetitions=4)
    param_est_3d = torch.tensor([1.0, 2.0])
    assert three_dim_leaf._broadcast_to_event_shape(param_est_3d).shape == (2, 2, 4)


def test_log_likelihood_raises_for_invalid_data_dim_and_event_shape_mismatch():
    leaf = TinyLeaf(scope=Scope([0]))
    with pytest.raises(ValueError):
        leaf.log_likelihood(torch.zeros((2, 1, 1)))

    leaf._event_shape = (2, 1, 1)
    with pytest.raises(RuntimeError):
        leaf.log_likelihood(torch.zeros((2, 1)))


def test_log_likelihood_interval_raises_when_cdf_is_missing_or_not_implemented():
    low = torch.full((3, 1), -1.0)
    high = torch.full((3, 1), 1.0)

    with pytest.raises(NotImplementedError):
        NoCdfLeaf(scope=Scope([0]))._log_likelihood_interval(low, high)

    with pytest.raises(NotImplementedError):
        RaisingCdfLeaf(scope=Scope([0]))._log_likelihood_interval(low, high)


def test_log_likelihood_dispatches_interval_evidence():
    leaf = TinyLeaf(scope=Scope([0]))
    evidence = IntervalEvidence(low=torch.full((2, 1), -0.5), high=torch.full((2, 1), 0.5))

    out = leaf.log_likelihood(evidence)

    assert out.shape == (2, 1, 1, 1)


def test_sample_conditional_mpe_raises_for_distribution_without_mode(monkeypatch):
    class DistNoMode:
        pass

    leaf = TinyLeaf(scope=Scope(query=[0], evidence=[1]), parameter_fn=SimpleParameterNet(value=0.0))
    monkeypatch.setattr(
        leaf,
        "conditional_distribution",
        lambda evidence, with_differentiable_sampling=False: DistNoMode(),
    )
    data = torch.tensor([[float("nan"), 1.0], [float("nan"), 2.0]])

    with pytest.raises(UnsupportedOperationError):
        leaf.sample(data=data, is_mpe=True)


def test_sample_conditional_mpe_mode_dim_variants(monkeypatch):
    class DistMode2D:
        mode = torch.tensor([[5.0], [6.0]])

    class DistMode3D:
        mode = torch.tensor([[[7.0]], [[8.0]]])

    data = torch.tensor([[float("nan"), 1.0], [float("nan"), 2.0]])
    leaf = TinyLeaf(scope=Scope(query=[0], evidence=[1]), parameter_fn=SimpleParameterNet(value=0.0))

    monkeypatch.setattr(
        leaf,
        "conditional_distribution",
        lambda evidence, with_differentiable_sampling=False: DistMode2D(),
    )
    out_2d = leaf.sample(data=data.clone(), is_mpe=True)
    torch.testing.assert_close(out_2d[:, 0], torch.tensor([5.0, 6.0]))

    monkeypatch.setattr(
        leaf,
        "conditional_distribution",
        lambda evidence, with_differentiable_sampling=False: DistMode3D(),
    )
    out_3d = leaf.sample(data=data.clone(), is_mpe=True)
    torch.testing.assert_close(out_3d[:, 0], torch.tensor([7.0, 8.0]))


def test_sample_conditional_paths_raise_for_invalid_sample_rank(monkeypatch):
    class DistBadModeRank:
        mode = torch.tensor([1.0, 2.0])

    class DistBadSampleRank:
        def sample(self, sample_shape):
            # Return a rank that must trip defensive shape checks in sampling.
            return torch.ones((1, 2, 1))

    data = torch.tensor([[float("nan"), 1.0], [float("nan"), 2.0]])
    leaf = TinyLeaf(scope=Scope(query=[0], evidence=[1]), parameter_fn=SimpleParameterNet(value=0.0))

    monkeypatch.setattr(
        leaf,
        "conditional_distribution",
        lambda evidence, with_differentiable_sampling=False: DistBadModeRank(),
    )
    with pytest.raises(ValueError):
        leaf.sample(data=data.clone(), is_mpe=True)

    monkeypatch.setattr(
        leaf,
        "conditional_distribution",
        lambda evidence, with_differentiable_sampling=False: DistBadSampleRank(),
    )
    with pytest.raises((ValueError, IndexError)):
        leaf.sample(data=data.clone(), is_mpe=False)


def test_resolve_scope_columns_and_slice_sampling_context_errors():
    leaf = TinyLeaf(scope=Scope([4, 5]))

    with pytest.raises(ShapeError):
        leaf._resolve_scope_columns(num_features=3)

    empty_scope_leaf = TinyLeaf(scope=Scope([]))
    assert empty_scope_leaf._resolve_scope_columns(num_features=3) == []

    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
    )
    with pytest.raises(ShapeError):
        leaf._slice_sampling_context(sampling_ctx=sampling_ctx, num_features=3, scope_cols=[0, 1])
