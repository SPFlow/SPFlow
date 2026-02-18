import numpy as np
import pytest
import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.measures._utils import (
    fork_rng,
    infer_discrete_domains,
    iter_modules,
    reduce_log_likelihood,
)
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape


class _ContainerModule(Module):
    def __init__(self, scope: Scope, inputs):
        super().__init__()
        self.scope = scope
        self.inputs = inputs
        self.in_shape = ModuleShape(len(scope.query), 1, 1)
        self.out_shape = ModuleShape(len(scope.query), 1, 1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([[Scope([q]) for q in self.scope.query]], dtype=object).T

    @property
    def device(self):
        return torch.device("cpu")

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:
        return torch.zeros((data.shape[0], len(self.scope.query), 1, 1), dtype=data.dtype, device=data.device)

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None) -> Tensor:
        if data is None:
            n = 1 if num_samples is None else num_samples
            return torch.zeros((n, len(self.scope.query)), dtype=torch.get_default_dtype())
        return data

    def _sample(self, data: Tensor, sampling_ctx, cache, is_mpe: bool = False) -> Tensor:
        del sampling_ctx
        del cache
        del is_mpe
        return data

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache=None):
        return self


class _InputFlakyModule(_ContainerModule):
    def __init__(self, scope: Scope):
        super().__init__(scope, inputs=[])
        self._n_gets = 0

    @property
    def inputs(self):
        self._n_gets += 1
        if self._n_gets == 1:
            return []
        raise AttributeError("inputs unavailable on second access")

    @inputs.setter
    def inputs(self, value):
        self._n_gets = 0


def test_reduce_log_likelihood_shapes_and_methods():
    ll2 = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    out2 = reduce_log_likelihood(ll2, channel_agg="first", repetition_agg="first")
    assert out2.shape == (2,)

    ll3 = torch.randn(3, 2, 4)
    out3 = reduce_log_likelihood(ll3, channel_agg="logsumexp", repetition_agg="first")
    assert out3.shape == (3,)

    ll4 = torch.randn(3, 2, 3, 2)
    out4 = reduce_log_likelihood(ll4, channel_agg="logmeanexp", repetition_agg="logsumexp")
    assert out4.shape == (3,)

    with pytest.raises(InvalidParameterError):
        reduce_log_likelihood(torch.randn(2), channel_agg="first", repetition_agg="first")

    with pytest.raises(InvalidParameterError):
        reduce_log_likelihood(torch.randn(2, 2, 2, 2), channel_agg="bad", repetition_agg="first")

    empty = torch.empty((0, 2, 1, 1))
    out_empty = reduce_log_likelihood(empty, channel_agg="first", repetition_agg="first")
    assert out_empty.shape == (0,)


def test_iter_modules_input_container_branches():
    leaf0 = Bernoulli(scope=Scope([0]), probs=torch.tensor([[[0.3]]]))
    leaf1 = Bernoulli(scope=Scope([1]), probs=torch.tensor([[[0.7]]]))

    list_root = _ContainerModule(Scope([0, 1]), inputs=[leaf0, leaf1])
    seen = list(iter_modules(list_root))
    assert leaf0 in seen and leaf1 in seen and list_root in seen

    flaky = _InputFlakyModule(Scope([0]))
    seen_flaky = list(iter_modules(flaky))
    assert seen_flaky == [flaky]

    bad = _ContainerModule(Scope([0]), inputs=123)
    with pytest.raises(UnsupportedOperationError):
        list(iter_modules(bad))


def test_infer_discrete_domains_success_and_errors():
    b = Bernoulli(scope=Scope([0]), probs=torch.tensor([[[0.5]]]))
    root = _ContainerModule(Scope([0]), inputs=[b])
    domains = infer_discrete_domains(root, Scope([0]))
    assert set(domains) == {0}
    torch.testing.assert_close(domains[0], torch.tensor([0.0, 1.0], dtype=domains[0].dtype))

    with pytest.raises(UnsupportedOperationError):
        infer_discrete_domains(root, Scope([1]))

    n = Normal(
        scope=Scope([0]),
        loc=torch.tensor([[[0.0]]], dtype=torch.get_default_dtype()),
        scale=torch.tensor([[[1.0]]], dtype=torch.get_default_dtype()),
    )
    root_with_normal = _ContainerModule(Scope([0]), inputs=[n])
    with pytest.raises(UnsupportedOperationError):
        infer_discrete_domains(root_with_normal, Scope([0]))

    c2 = Categorical(scope=Scope([0]), K=2, probs=torch.tensor([[[[0.5, 0.5]]]]))
    c3 = Categorical(scope=Scope([0]), K=3, probs=torch.tensor([[[[0.2, 0.3, 0.5]]]]))
    inconsistent_root = _ContainerModule(Scope([0]), inputs=[c2, c3])
    with pytest.raises(UnsupportedOperationError):
        infer_discrete_domains(inconsistent_root, Scope([0]))

    bad_k = Categorical(scope=Scope([0]), K=2, probs=torch.tensor([[[[0.5, 0.5]]]]))
    bad_k.K = 0
    bad_k_root = _ContainerModule(Scope([0]), inputs=[bad_k])
    with pytest.raises(UnsupportedOperationError):
        infer_discrete_domains(bad_k_root, Scope([0]))


def test_fork_rng_seed_none_and_cuda_device_handling(monkeypatch):
    ctx = fork_rng(None, torch.device("cpu"))
    with ctx:
        pass

    captured = {}

    class _DummyCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def _fake_fork_rng(*, devices, enabled):
        captured["devices"] = devices
        captured["enabled"] = enabled
        return _DummyCtx()

    monkeypatch.setattr(torch.random, "fork_rng", _fake_fork_rng)
    _ = fork_rng(7, torch.device("cuda:0"))
    assert captured["devices"] == [0]
    assert captured["enabled"] is True
