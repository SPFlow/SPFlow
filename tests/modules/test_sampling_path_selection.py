"""Tests for hard vs differentiable sampling dispatch on ``Module``."""

import inspect

import numpy as np
import pytest
import torch

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import DifferentiableSamplingContext, SamplingContext


class _PathProbe(Module):
    def __init__(self) -> None:
        super().__init__()
        self.scope = Scope([0])
        self.in_shape = ModuleShape(1, 1, 1)
        self.out_shape = ModuleShape(1, 1, 1)
        self.sample_calls = 0
        self.rsample_calls = 0
        self.last_sampling_ctx = None

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([0])], dtype=object).reshape(1, 1)

    def log_likelihood(self, data: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        del cache
        return torch.zeros((data.shape[0], 1, 1, 1), dtype=data.dtype, device=data.device)

    def _sample(
        self,
        data: torch.Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> torch.Tensor:
        del cache
        del is_mpe
        self.sample_calls += 1
        self.last_sampling_ctx = sampling_ctx
        out = data.clone()
        out[torch.isnan(out)] = 1.0
        return out

    def _rsample(
        self,
        data: torch.Tensor,
        sampling_ctx: DifferentiableSamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> torch.Tensor:
        del cache
        del is_mpe
        self.rsample_calls += 1
        self.last_sampling_ctx = sampling_ctx
        out = data.clone()
        out[torch.isnan(out)] = 2.0
        return out

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache: Cache | None = None):
        del marg_rvs
        del prune
        del cache
        return self


class _NoDifferentiableProbe(Module):
    def __init__(self) -> None:
        super().__init__()
        self.scope = Scope([0])
        self.in_shape = ModuleShape(1, 1, 1)
        self.out_shape = ModuleShape(1, 1, 1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([0])], dtype=object).reshape(1, 1)

    def log_likelihood(self, data: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        del cache
        return torch.zeros((data.shape[0], 1, 1, 1), dtype=data.dtype, device=data.device)

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
        return data

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache: Cache | None = None):
        del marg_rvs
        del prune
        del cache
        return self


def test_sample_dispatches_to_hard_path_only():
    node = _PathProbe()

    out = node.sample(num_samples=3)

    assert node.sample_calls == 1
    assert node.rsample_calls == 0
    assert isinstance(node.last_sampling_ctx, SamplingContext)
    assert torch.equal(out, torch.ones((3, 1)))


def test_rsample_dispatches_to_differentiable_path_only():
    node = _PathProbe()

    out = node.rsample(
        num_samples=4,
        diff_method="gumbel",
        hard=True,
        temperature_sums=0.7,
        temperature_leaves=0.9,
    )

    assert node.sample_calls == 0
    assert node.rsample_calls == 1
    assert isinstance(node.last_sampling_ctx, DifferentiableSamplingContext)
    assert node.last_sampling_ctx.diff_method == "gumbel"
    assert node.last_sampling_ctx.hard is True
    assert node.last_sampling_ctx.temperature_sums == pytest.approx(0.7)
    assert node.last_sampling_ctx.temperature_leaves == pytest.approx(0.9)
    assert torch.equal(out, torch.full((4, 1), 2.0))


def test_rsample_rejects_invalid_diff_method():
    node = _PathProbe()

    with pytest.raises(InvalidParameterError, match="diff_method"):
        node.rsample(num_samples=2, diff_method="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "temperature_sums, temperature_leaves", [(0.0, 1.0), (1.0, 0.0), (-0.5, 1.0), (1.0, -2.0)]
)
def test_rsample_rejects_non_positive_temperatures(temperature_sums: float, temperature_leaves: float):
    node = _PathProbe()

    with pytest.raises(InvalidParameterError, match="must be > 0"):
        node.rsample(
            num_samples=1,
            temperature_sums=temperature_sums,
            temperature_leaves=temperature_leaves,
        )


def test_rsample_public_api_has_no_tau_parameter():
    sig = inspect.signature(Module.rsample)
    assert "tau" not in sig.parameters


def test_default_module_rsample_is_fail_fast():
    node = _NoDifferentiableProbe()

    with pytest.raises(UnsupportedOperationError, match="does not implement differentiable sampling"):
        node.rsample(num_samples=2)
