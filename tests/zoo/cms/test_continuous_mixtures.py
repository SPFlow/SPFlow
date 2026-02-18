import numpy as np
import pytest
import torch
import torch.nn as nn

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.zoo.cms import (
    JointLogLikelihood,
    LatentOptimizationConfig,
    learn_continuous_mixture_cltree,
    learn_continuous_mixture_factorized,
)
from spflow.zoo.cms import continuous_mixtures as cms_mod
from spflow.zoo.cms.continuous_mixtures import (
    _broadcast_component_weights,
    _compile_factorized,
    _factorized_component_ll,
    _iter_minibatches,
    _latent_opt_cltree,
    _latent_opt_factorized,
    _make_sum_weights,
    _mixture_log_likelihood_from_component_ll,
    _to_device_dtype,
)
from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.leaves import CLTree
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext


def test_joint_log_likelihood_wrapper_reduces_feature_axis():
    K = 3
    scope = Scope([0, 1, 2])
    parents = torch.tensor([-1, 0, 1], dtype=torch.long)
    log_cpt = torch.rand((3, 1, 1, K, K), dtype=torch.float64)
    log_cpt = log_cpt / log_cpt.sum(dim=3, keepdim=True).clamp_min(1e-12)
    log_cpt = log_cpt.clamp_min(1e-12).log()

    base = CLTree(scope=scope, out_channels=1, num_repetitions=1, K=K, parents=parents, log_cpt=log_cpt)
    wrapped = JointLogLikelihood(base)

    data = base.sample(num_samples=5).to(torch.float64)
    ll_base = base.log_likelihood(data)
    ll_wrapped = wrapped.log_likelihood(data)

    assert ll_base.shape[1] == 3
    assert ll_wrapped.shape[1] == 1
    torch.testing.assert_close(ll_wrapped.squeeze(1), ll_base.sum(dim=1), rtol=1e-6, atol=1e-6)


def test_learn_continuous_mixture_factorized_bernoulli_smoke_with_lo():
    torch.manual_seed(0)
    # Simple independent Bernoulli data with NaNs sprinkled in.
    N, F = 200, 6
    probs = torch.linspace(0.1, 0.9, F)
    data = torch.bernoulli(probs.expand(N, F)).to(torch.float32)
    data[0:10, 0] = float("nan")

    model = learn_continuous_mixture_factorized(
        data,
        leaf="bernoulli",
        latent_dim=2,
        num_points_train=16,
        num_points_eval=16,
        num_epochs=5,
        batch_size=64,
        lr=1e-3,
        seed=0,
        lo=LatentOptimizationConfig(enabled=True, num_points=8, num_epochs=5, batch_size=64, lr=5e-2),
    )

    ll = model.log_likelihood(data)
    assert torch.isfinite(ll).all()
    assert ll.shape[0] == N


def test_learn_continuous_mixture_factorized_categorical_smoke():
    torch.manual_seed(0)
    N, F, K = 150, 5, 4
    # Independent categorical.
    logits = torch.randn(F, K)
    probs = torch.softmax(logits, dim=-1)
    x = torch.multinomial(probs, num_samples=N, replacement=True).T.contiguous().to(torch.float32)

    model = learn_continuous_mixture_factorized(
        x,
        leaf="categorical",
        num_cats=K,
        latent_dim=2,
        num_points_train=16,
        num_points_eval=16,
        num_epochs=3,
        batch_size=64,
        lr=1e-3,
        seed=0,
        lo=None,
    )
    ll = model.log_likelihood(x)
    assert torch.isfinite(ll).all()


def test_learn_continuous_mixture_factorized_normal_smoke():
    torch.manual_seed(0)
    N, F = 200, 4
    loc = torch.linspace(-1.0, 1.0, F)
    scale = torch.linspace(0.5, 1.0, F)
    data = torch.randn(N, F) * scale + loc
    data = data.to(torch.float32)

    model = learn_continuous_mixture_factorized(
        data,
        leaf="normal",
        latent_dim=2,
        num_points_train=16,
        num_points_eval=16,
        num_epochs=3,
        batch_size=64,
        lr=1e-3,
        seed=0,
        lo=LatentOptimizationConfig(enabled=True, num_points=8, num_epochs=3, batch_size=64, lr=5e-2),
    )
    ll = model.log_likelihood(data)
    assert torch.isfinite(ll).all()


def test_learn_continuous_mixture_cltree_smoke_with_lo():
    torch.manual_seed(0)
    K = 3
    # Use a fixed simple CLTree to generate data.
    scope = Scope([0, 1, 2])
    parents = torch.tensor([-1, 0, 1], dtype=torch.long)
    log_cpt = torch.rand((3, 1, 1, K, K), dtype=torch.float64)
    log_cpt = log_cpt / log_cpt.sum(dim=3, keepdim=True).clamp_min(1e-12)
    log_cpt = log_cpt.clamp_min(1e-12).log()
    true_model = CLTree(scope=scope, out_channels=1, num_repetitions=1, K=K, parents=parents, log_cpt=log_cpt)

    data = true_model.sample(num_samples=250).to(torch.float32)

    model = learn_continuous_mixture_cltree(
        data,
        leaf="categorical",
        num_cats=K,
        latent_dim=2,
        num_points_train=16,
        num_points_eval=16,
        num_epochs=5,
        batch_size=64,
        lr=1e-3,
        seed=0,
        lo=LatentOptimizationConfig(enabled=True, num_points=8, num_epochs=5, batch_size=64, lr=5e-2),
    )

    ll = model.log_likelihood(data)
    assert torch.isfinite(ll).all()
    assert ll.shape[1] == 1


class _DummyModule(Module):
    def __init__(self, features: int = 2, repetitions: int = 2, marginalize_result=None):
        super().__init__()
        self.scope = Scope(list(range(features)))
        self.in_shape = ModuleShape(features, 1, 1)
        self.out_shape = ModuleShape(features, 1, repetitions)
        self._marginalize_result = marginalize_result
        self.sample_calls = 0

    @property
    def feature_to_scope(self):
        return self._feature_to_scope

    @feature_to_scope.setter
    def feature_to_scope(self, value):
        self._feature_to_scope = value

    def log_likelihood(self, data, cache: Cache | None = None):
        return torch.zeros(
            (data.shape[0], self.out_shape.features, self.out_shape.channels, self.out_shape.repetitions),
            dtype=data.dtype,
            device=data.device,
        )

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None):
        self.sample_calls += 1
        if data is None:
            n = 1 if num_samples is None else num_samples
            data = torch.zeros((n, self.out_shape.features), dtype=torch.get_default_dtype())
        return data

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
        self.sample_calls += 1
        return data

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache: Cache | None = None):
        return self._marginalize_result


def test_joint_log_likelihood_feature_to_scope_and_delegation_paths():
    base = _DummyModule(features=3, repetitions=2)
    base.feature_to_scope = np.array(
        [
            [Scope([0]), Scope([0])],
            [Scope([1]), Scope([1])],
            [Scope([2]), Scope([2])],
        ],
        dtype=object,
    )
    wrapped = JointLogLikelihood(base)

    f2s = wrapped.feature_to_scope
    assert f2s.shape == (1, 2)
    assert f2s[0, 0] == Scope([0, 1, 2])
    assert f2s[0, 1] == Scope([0, 1, 2])

    out = wrapped.sample(num_samples=4)
    assert out.shape == (4, 3)
    assert base.sample_calls == 1

    child_none = _DummyModule(features=2, repetitions=1, marginalize_result=None)
    wrapped_none = JointLogLikelihood(child_none)
    assert wrapped_none.marginalize([0]) is None

    child_single = _DummyModule(features=2, repetitions=1)
    child_single.out_shape = ModuleShape(1, 1, 1)
    parent_single = _DummyModule(features=2, repetitions=1, marginalize_result=child_single)
    wrapped_single = JointLogLikelihood(parent_single)
    assert wrapped_single.marginalize([0]) is child_single

    child_multi = _DummyModule(features=2, repetitions=1)
    parent_multi = _DummyModule(features=2, repetitions=1, marginalize_result=child_multi)
    wrapped_multi = JointLogLikelihood(parent_multi)
    out_multi = wrapped_multi.marginalize([0])
    assert isinstance(out_multi, JointLogLikelihood)


def test_cms_helpers_and_weight_validation_paths():
    x = torch.arange(6, dtype=torch.float32).view(3, 2)
    x64 = _to_device_dtype(x, device=x.device, dtype=torch.float64)
    assert x64.dtype == torch.float64
    assert x64.device == x.device

    gen = torch.Generator(device=x.device)
    gen.manual_seed(7)
    batches = list(_iter_minibatches(x64, batch_size=99, generator=gen))
    assert len(batches) == 1
    torch.testing.assert_close(batches[0], x64)

    w = _make_sum_weights(num_components=3, num_features=2, device=x.device, dtype=x.dtype)
    assert w.shape == (2, 3, 1, 1)
    torch.testing.assert_close(w.sum(dim=1), torch.ones((2, 1, 1), dtype=x.dtype))

    wb = _broadcast_component_weights(weights=torch.tensor([0.2, 0.3, 0.5]), num_features=2)
    assert wb.shape == (2, 3, 1, 1)

    with pytest.raises(InvalidParameterError):
        _broadcast_component_weights(weights=torch.ones(2, 2), num_features=2)
    with pytest.raises(InvalidParameterError):
        _broadcast_component_weights(weights=torch.tensor([0.5, 0.5, 0.0]), num_features=2)
    with pytest.raises(InvalidParameterError):
        _broadcast_component_weights(weights=torch.tensor([0.5, 0.6]), num_features=2)

    component_ll = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    mix = _mixture_log_likelihood_from_component_ll(component_ll, torch.tensor([0.4, 0.6]))
    assert mix.shape == (2,)
    assert torch.isfinite(mix).all()

    with pytest.raises(InvalidParameterError):
        _mixture_log_likelihood_from_component_ll(torch.zeros(2, 2, 1), torch.tensor([0.5, 0.5]))
    with pytest.raises(InvalidParameterError):
        _mixture_log_likelihood_from_component_ll(torch.zeros(2, 2), torch.tensor([[0.5, 0.5]]))


def test_factorized_component_ll_validation_branches():
    with pytest.raises(InvalidParameterError):
        _factorized_component_ll(
            data=torch.tensor([0.0, 1.0]),
            leaf="bernoulli",
            decoder_out=torch.zeros(2, 2),
            num_cats=None,
            normal_eps=1e-4,
        )

    with pytest.raises(InvalidParameterError):
        _factorized_component_ll(
            data=torch.tensor([[0.5, 1.0]]),
            leaf="bernoulli",
            decoder_out=torch.zeros(2, 2),
            num_cats=None,
            normal_eps=1e-4,
        )
    with pytest.raises(InvalidParameterError):
        _factorized_component_ll(
            data=torch.tensor([[0.0, 2.0]]),
            leaf="bernoulli",
            decoder_out=torch.zeros(2, 2),
            num_cats=None,
            normal_eps=1e-4,
        )

    with pytest.raises(InvalidParameterError):
        _factorized_component_ll(
            data=torch.tensor([[0.0, 1.0]]),
            leaf="categorical",
            decoder_out=torch.zeros(2, 4),
            num_cats=None,
            normal_eps=1e-4,
        )
    with pytest.raises(InvalidParameterError):
        _factorized_component_ll(
            data=torch.tensor([[0.2, 1.0]]),
            leaf="categorical",
            decoder_out=torch.zeros(2, 6),
            num_cats=3,
            normal_eps=1e-4,
        )
    with pytest.raises(InvalidParameterError):
        _factorized_component_ll(
            data=torch.tensor([[0.0, 3.0]]),
            leaf="categorical",
            decoder_out=torch.zeros(2, 6),
            num_cats=3,
            normal_eps=1e-4,
        )

    normal_ll = _factorized_component_ll(
        data=torch.tensor([[0.0, float("nan")], [1.0, 2.0]]),
        leaf="normal",
        decoder_out=torch.zeros(2, 4),
        num_cats=None,
        normal_eps=1e-4,
    )
    assert normal_ll.shape == (2, 2)
    assert torch.isfinite(normal_ll).all()

    with pytest.raises(InvalidParameterError):
        _factorized_component_ll(
            data=torch.zeros(2, 2),
            leaf="bad",  # type: ignore[arg-type]
            decoder_out=torch.zeros(2, 2),
            num_cats=None,
            normal_eps=1e-4,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"leaf": "bernoulli", "latent_dim": 0}, "latent_dim"),
        ({"leaf": "bernoulli", "num_points_train": 0}, "num_points_train"),
        ({"leaf": "bernoulli", "num_epochs": 0}, "num_epochs"),
        ({"leaf": "bernoulli", "batch_size": 0}, "batch_size"),
        ({"leaf": "bernoulli", "lr": 0.0}, "lr"),
        ({"leaf": "bernoulli", "patience": -1}, "patience"),
        ({"leaf": "bernoulli", "num_points_eval": 0}, "num_points_eval"),
        ({"leaf": "categorical"}, "num_cats"),
        ({"leaf": "invalid"}, "Unsupported leaf type"),
    ],
)
def test_learn_continuous_mixture_factorized_validation(kwargs, message):
    data = torch.zeros(10, 2)
    with pytest.raises(InvalidParameterError):
        learn_continuous_mixture_factorized(data, **kwargs)  # type: ignore[arg-type]


def test_learn_continuous_mixture_factorized_rejects_non_2d_data():
    with pytest.raises(InvalidParameterError):
        learn_continuous_mixture_factorized(torch.zeros(10), leaf="bernoulli")


def test_compile_factorized_error_paths():
    z = torch.zeros(2, 2)

    class _Decoder(nn.Module):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return torch.zeros((z.shape[0], 4), dtype=z.dtype, device=z.device)

    dec = _Decoder()
    with pytest.raises(InvalidParameterError):
        _compile_factorized(
            decoder=dec,
            leaf="categorical",
            z=z,
            weights=torch.tensor([0.5, 0.5]),
            num_features=2,
            num_cats=None,
            normal_eps=1e-4,
            device=z.device,
            dtype=z.dtype,
        )

    with pytest.raises(InvalidParameterError):
        _compile_factorized(
            decoder=dec,
            leaf="bad",  # type: ignore[arg-type]
            z=z,
            weights=torch.tensor([0.5, 0.5]),
            num_features=2,
            num_cats=2,
            normal_eps=1e-4,
            device=z.device,
            dtype=z.dtype,
        )


def test_latent_opt_helpers_early_stop_branches():
    class _FlatDecoder(nn.Module):
        def __init__(self, out_dim: int):
            super().__init__()
            self.out_dim = out_dim

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            base = torch.zeros((z.shape[0], self.out_dim), dtype=z.dtype, device=z.device)
            return base + z.sum(dim=1, keepdim=True) * 0.0

    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    cfg = LatentOptimizationConfig(num_points=3, num_epochs=4, batch_size=2, lr=1e-2, patience=1, seed=1)
    z_opt, w_opt = _latent_opt_factorized(
        data=x,
        val_data=x,
        leaf="bernoulli",
        decoder=_FlatDecoder(out_dim=2),
        latent_dim=2,
        num_cats=None,
        normal_eps=1e-4,
        cfg=cfg,
    )
    assert z_opt.shape == (cfg.num_points, 2)
    assert w_opt.shape == (cfg.num_points,)

    parents = torch.tensor([-1, 0], dtype=torch.long)

    def _decode_log_cpt(z: torch.Tensor) -> torch.Tensor:
        I = z.shape[0]
        log_half = torch.log(torch.tensor(0.5, dtype=z.dtype, device=z.device))
        zeros = torch.zeros((I, 2, 2, 2), dtype=z.dtype, device=z.device) + z.sum() * 0.0
        return zeros + log_half

    z_opt_clt, w_opt_clt = _latent_opt_cltree(
        data=x,
        val_data=x,
        decoder=nn.Identity(),
        decode_log_cpt=_decode_log_cpt,
        parents=parents,
        root=0,
        K=2,
        latent_dim=2,
        cfg=cfg,
    )
    assert z_opt_clt.shape == (cfg.num_points, 2)
    assert w_opt_clt.shape == (cfg.num_points,)


def test_cltree_validation_branches_and_no_lo_path():
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    with pytest.raises(UnsupportedOperationError):
        learn_continuous_mixture_cltree(data, leaf="normal")  # type: ignore[arg-type]
    with pytest.raises(InvalidParameterError):
        learn_continuous_mixture_cltree(torch.zeros(4), leaf="bernoulli")
    with pytest.raises(InvalidParameterError):
        learn_continuous_mixture_cltree(torch.tensor([[0.0, float("nan")]]), leaf="bernoulli")
    with pytest.raises(InvalidParameterError):
        learn_continuous_mixture_cltree(torch.tensor([[0.1, 1.0]]), leaf="bernoulli")
    with pytest.raises(InvalidParameterError):
        learn_continuous_mixture_cltree(data, leaf="categorical", num_cats=1)
    with pytest.raises(InvalidParameterError):
        learn_continuous_mixture_cltree(torch.tensor([[0.0, 2.0]]), leaf="bernoulli")

    model = learn_continuous_mixture_cltree(
        data,
        leaf="bernoulli",
        latent_dim=2,
        num_points_train=4,
        num_points_eval=4,
        num_epochs=2,
        batch_size=2,
        lr=1e-3,
        seed=0,
        val_data=data.clone(),
        lo=None,
    )
    ll = model.log_likelihood(data)
    assert ll.shape[1] == 1
    assert torch.isfinite(ll).all()


def test_factorized_early_stopping_break_and_val_data_cast(monkeypatch):
    def _flat_mix(component_ll: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return component_ll.mean(dim=0) * 0.0

    monkeypatch.setattr(cms_mod, "_mixture_log_likelihood_from_component_ll", _flat_mix)
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    model = learn_continuous_mixture_factorized(
        data,
        leaf="bernoulli",
        latent_dim=2,
        num_points_train=4,
        num_points_eval=4,
        num_epochs=4,
        batch_size=2,
        lr=1e-3,
        patience=1,
        val_data=data.clone(),
        dtype=torch.float64,
        lo=None,
    )
    ll = model.log_likelihood(data.to(torch.float64))
    assert torch.isfinite(ll).all()


def test_cltree_early_stopping_break_branch(monkeypatch):
    def _flat_mix(component_ll: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return component_ll.mean(dim=0) * 0.0

    monkeypatch.setattr(cms_mod, "_mixture_log_likelihood_from_component_ll", _flat_mix)
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    model = learn_continuous_mixture_cltree(
        data,
        leaf="bernoulli",
        latent_dim=2,
        num_points_train=4,
        num_points_eval=4,
        num_epochs=4,
        batch_size=2,
        lr=1e-3,
        patience=1,
        val_data=data.clone(),
        lo=None,
    )
    ll = model.log_likelihood(data)
    assert torch.isfinite(ll).all()
