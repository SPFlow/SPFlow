import numpy as np
import pytest
import torch

from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape


class _ChildRecorder(Module):
    def __init__(self):
        super().__init__()
        self.scope = Scope([0])
        self.in_shape = ModuleShape(1, 1, 1)
        self.out_shape = ModuleShape(1, 1, 1)
        self.em_calls = 0
        self.mle_calls = 0

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([0])], dtype=object).reshape(1, 1)

    def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:
        return torch.zeros((data.shape[0], 1, 1, 1), dtype=data.dtype, device=data.device)

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
        data = self._prepare_sample_data(num_samples, data)
        data[:] = torch.nan_to_num(data, nan=0.0)
        return data

    def expectation_maximization(self, data: torch.Tensor, bias_correction: bool = True, cache=None) -> None:
        self.em_calls += 1

    def maximum_likelihood_estimation(
        self,
        data: torch.Tensor,
        weights: torch.Tensor | None = None,
        bias_correction: bool = True,
        nan_strategy: str = "ignore",
        cache=None,
    ) -> None:
        self.mle_calls += 1

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache=None):
        return self


class _ParentDispatch(Module):
    def __init__(self, inputs):
        super().__init__()
        self.scope = Scope([0])
        self.in_shape = ModuleShape(1, 1, 1)
        self.out_shape = ModuleShape(1, 1, 1)
        self.inputs = inputs

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([0])], dtype=object).reshape(1, 1)

    def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:
        return torch.zeros((data.shape[0], 1, 1, 1), dtype=data.dtype, device=data.device)

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
        data = self._prepare_sample_data(num_samples, data)
        data[:] = torch.nan_to_num(data, nan=0.0)
        return data

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache=None):
        return self


class _SuperPassProbe(Module):
    def __init__(self):
        super().__init__()
        self.scope = Scope([0])
        self.in_shape = ModuleShape(1, 1, 1)
        self.out_shape = ModuleShape(1, 1, 1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        Module.feature_to_scope.fget(self)  # type: ignore[misc]
        return np.array([Scope([0])], dtype=object).reshape(1, 1)

    def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:
        Module.log_likelihood(self, data, cache=cache)
        return torch.zeros((data.shape[0], 1, 1, 1), dtype=data.dtype, device=data.device)

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
        Module.sample(
            self, num_samples=num_samples, data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx
        )
        data = self._prepare_sample_data(num_samples, data)
        data[:] = torch.nan_to_num(data, nan=0.0)
        return data

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache=None):
        Module.marginalize(self, marg_rvs=marg_rvs, prune=prune, cache=cache)
        return self


def test_prepare_sample_data_mismatch_raises():
    node = _ChildRecorder()
    data = torch.zeros((2, 1))
    with pytest.raises(ValueError):
        node._prepare_sample_data(num_samples=3, data=data)


def test_prepare_sample_data_defaults_to_one_sample():
    node = _ChildRecorder()
    data = node._prepare_sample_data(num_samples=None, data=None)
    assert data.shape == (1, 1)


def test_sample_with_evidence_creates_cache_and_delegates_sample():
    node = _ChildRecorder()
    evidence = torch.tensor([[float("nan")]])
    out = node.sample_with_evidence(evidence=evidence, cache=None)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()


def test_expectation_maximization_dispatches_for_list_inputs():
    c1 = _ChildRecorder()
    c2 = _ChildRecorder()
    parent = _ParentDispatch(inputs=[c1, c2])

    parent.expectation_maximization(torch.zeros((2, 1)), cache=None)
    assert c1.em_calls == 1
    assert c2.em_calls == 1


def test_expectation_and_mle_dispatch_for_single_input():
    child = _ChildRecorder()
    parent = _ParentDispatch(inputs=child)

    parent.expectation_maximization(torch.zeros((2, 1)), cache=None)
    parent.maximum_likelihood_estimation(torch.zeros((2, 1)), cache=None)

    assert child.em_calls == 1
    assert child.mle_calls == 1


def test_forward_and_probability_delegate_to_log_likelihood():
    node = _ChildRecorder()
    x = torch.zeros((3, 1))

    ll = node.forward(x)
    p = node.probability(x)

    assert ll.shape == (3, 1, 1, 1)
    assert torch.allclose(p, torch.ones_like(ll))


def test_mpe_and_structure_stats_and_extra_vis_info_paths():
    child = _ChildRecorder()
    parent = _ParentDispatch(inputs=[child, _ChildRecorder()])

    x = torch.full((2, 1), float("nan"))
    mpe = child.mpe(data=x)
    assert torch.isfinite(mpe).all()

    parent.maximum_likelihood_estimation(torch.zeros((2, 1)), cache=None)
    assert child.mle_calls == 1

    assert child._extra_vis_info() is None
    stats = child.print_structure_stats()
    assert isinstance(stats, str)


def test_super_pass_probe_executes_abstract_super_bodies():
    probe = _SuperPassProbe()
    x = torch.zeros((2, 1))

    _ = probe.feature_to_scope
    probe.log_likelihood(x)
    probe.sample(num_samples=2)
    probe.marginalize([0])
