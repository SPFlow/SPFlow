import math

import pytest
import torch

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.products.product import Product
from spflow.modules.sos.socs import SOCS
from spflow.modules.sos.socs import _signed_eval
from spflow.modules.sums.signed_sum import SignedSum
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot


def _make_normal_component(mu: float, sigma: float) -> Normal:
    loc = torch.tensor([[[mu]]], dtype=torch.get_default_dtype())
    scale = torch.tensor([[[sigma]]], dtype=torch.get_default_dtype())
    return Normal(scope=Scope([0]), out_channels=1, num_repetitions=1, loc=loc, scale=scale)


def test_socs_sample_rejects_differentiable_routing():
    model = SOCS([_make_normal_component(mu=0.0, sigma=1.0)])
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(torch.zeros((3, 1), dtype=torch.long), dim=-1, dim_size=1),
        mask=torch.ones((3, 1), dtype=torch.bool),
        repetition_index=to_one_hot(torch.zeros((3,), dtype=torch.long), dim=-1, dim_size=1),
        is_differentiable=True,
    )
    with pytest.raises(UnsupportedOperationError, match="differentiable routing"):
        model._sample(
            data=torch.full((3, 1), float("nan")),
            sampling_ctx=sampling_ctx,
            cache=Cache(),
        )


def test_socs_single_normal_matches_known_squared_density_and_Z():
    sigma = 1.5
    mu = -0.2
    comp = _make_normal_component(mu=mu, sigma=sigma)
    model = SOCS([comp])

    x = torch.randn(11, 1)
    cache = Cache()
    ll = model.log_likelihood(x, cache=cache)  # (B,1,1,1)

    # Target distribution is Normal(mu, sigma/sqrt(2)).
    target = torch.distributions.Normal(
        loc=torch.tensor(mu, dtype=x.dtype, device=x.device),
        scale=torch.tensor(sigma / math.sqrt(2.0), dtype=x.dtype, device=x.device),
    )
    expected = target.log_prob(x.squeeze(1))

    torch.testing.assert_close(ll.squeeze(-1).squeeze(-1).squeeze(1), expected, rtol=1e-4, atol=1e-4)

    # Z = ∫ N(x;mu,sigma)^2 dx = 1 / (2*sqrt(pi)*sigma)
    logZ = cache.extras["socs_logZ"]
    expected_logZ = math.log(1.0 / (2.0 * math.sqrt(math.pi) * sigma))
    torch.testing.assert_close(
        logZ[0, 0, 0],
        torch.tensor(expected_logZ, dtype=logZ.dtype, device=logZ.device),
        rtol=0.0,
        atol=0.0,
    )


def test_socs_two_normals_matches_expected_density_and_Z():
    comp1 = _make_normal_component(mu=-0.7, sigma=1.2)
    comp2 = _make_normal_component(mu=0.9, sigma=0.6)
    model = SOCS([comp1, comp2])

    x = torch.linspace(-2.0, 2.0, steps=21).unsqueeze(1)
    cache = Cache()
    ll = model.log_likelihood(x, cache=cache).squeeze(-1).squeeze(-1).squeeze(1)

    def normal_pdf(xv: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
        return torch.exp(
            torch.distributions.Normal(
                loc=torch.tensor(mu, dtype=xv.dtype, device=xv.device),
                scale=torch.tensor(sigma, dtype=xv.dtype, device=xv.device),
            ).log_prob(xv)
        )

    f1 = normal_pdf(x.squeeze(1), mu=-0.7, sigma=1.2)
    f2 = normal_pdf(x.squeeze(1), mu=0.9, sigma=0.6)
    num = f1.pow(2) + f2.pow(2)

    z1 = 1.0 / (2.0 * math.sqrt(math.pi) * 1.2)
    z2 = 1.0 / (2.0 * math.sqrt(math.pi) * 0.6)
    z = z1 + z2
    expected = torch.log(num) - math.log(z)

    torch.testing.assert_close(ll, expected, rtol=2e-4, atol=2e-4)

    logZ = cache.extras["socs_logZ"]
    torch.testing.assert_close(
        logZ[0, 0, 0],
        torch.tensor(math.log(z), dtype=logZ.dtype, device=logZ.device),
        rtol=0.0,
        atol=1e-6,
    )


def _bernoulli_probs(p: float) -> tuple[float, float]:
    return (1.0 - p, p)


def test_inner_product_sum_bernoulli_matches_exact_enumeration():
    p1 = 0.2
    p2 = 0.75
    w1 = 0.3
    w2 = 0.7

    probs1 = torch.tensor([[[p1]]], dtype=torch.get_default_dtype())
    probs2 = torch.tensor([[[p2]]], dtype=torch.get_default_dtype())

    b1 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=probs1)
    b2 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=probs2)
    mix = Sum(inputs=[b1, b2], weights=torch.tensor([[[[w1]], [[w2]]]]))

    # Exact enumeration over {0,1}
    p10, p11 = _bernoulli_probs(p1)
    p20, p21 = _bernoulli_probs(p2)
    m0 = w1 * p10 + w2 * p20
    m1 = w1 * p11 + w2 * p21
    z = m0 * m0 + m1 * m1

    from spflow.utils.inner_product import log_self_inner_product_scalar

    logZ = log_self_inner_product_scalar(mix)
    torch.testing.assert_close(
        logZ, torch.tensor(math.log(z), dtype=logZ.dtype, device=logZ.device), rtol=1e-6, atol=1e-6
    )


def test_inner_product_signed_sum_bernoulli_matches_exact_enumeration():
    p1 = 0.1
    p2 = 0.9
    w1 = 1.0
    w2 = -0.4

    probs1 = torch.tensor([[[p1]]], dtype=torch.get_default_dtype())
    probs2 = torch.tensor([[[p2]]], dtype=torch.get_default_dtype())
    b1 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=probs1)
    b2 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=probs2)
    node = SignedSum(
        inputs=[b1, b2], out_channels=1, num_repetitions=1, weights=torch.tensor([[[[w1]], [[w2]]]])
    )

    # Exact enumeration over {0,1}
    p10, p11 = _bernoulli_probs(p1)
    p20, p21 = _bernoulli_probs(p2)
    c0 = w1 * p10 + w2 * p20
    c1 = w1 * p11 + w2 * p21
    z = c0 * c0 + c1 * c1

    from spflow.utils.inner_product import log_self_inner_product_scalar

    logZ = log_self_inner_product_scalar(node)
    torch.testing.assert_close(
        logZ, torch.tensor(math.log(z), dtype=logZ.dtype, device=logZ.device), rtol=1e-6, atol=1e-6
    )


def test_socs_signed_component_matches_exact_bernoulli_enumeration():
    p1 = 0.25
    p2 = 0.8
    w1 = 0.9
    w2 = -0.2

    probs1 = torch.tensor([[[p1]]], dtype=torch.get_default_dtype())
    probs2 = torch.tensor([[[p2]]], dtype=torch.get_default_dtype())
    b1 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=probs1)
    b2 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=probs2)
    comp = SignedSum(
        inputs=[b1, b2], out_channels=1, num_repetitions=1, weights=torch.tensor([[[[w1]], [[w2]]]])
    )
    model = SOCS([comp])

    # Exact values over domain {0,1}
    p10, p11 = _bernoulli_probs(p1)
    p20, p21 = _bernoulli_probs(p2)
    c0 = w1 * p10 + w2 * p20
    c1 = w1 * p11 + w2 * p21
    z = c0 * c0 + c1 * c1

    x = torch.tensor([[0.0], [1.0]], dtype=torch.get_default_dtype())
    ll = model.log_likelihood(x).squeeze(-1).squeeze(-1).squeeze(1)
    expected = torch.tensor(
        [math.log((c0 * c0) / z), math.log((c1 * c1) / z)],
        dtype=ll.dtype,
        device=ll.device,
    )
    torch.testing.assert_close(ll, expected, rtol=1e-6, atol=1e-6)


def test_inner_product_categorical_matches_exact_enumeration():
    # Single-variable categorical, mixture of two categoricals.
    K = 3
    w1 = 0.4
    w2 = 0.6

    p1 = torch.tensor([[[[0.1, 0.7, 0.2]]]], dtype=torch.get_default_dtype())  # (F,C,R,K)
    p2 = torch.tensor([[[[0.3, 0.3, 0.4]]]], dtype=torch.get_default_dtype())
    c1 = Categorical(scope=Scope([0]), out_channels=1, num_repetitions=1, K=K, probs=p1)
    c2 = Categorical(scope=Scope([0]), out_channels=1, num_repetitions=1, K=K, probs=p2)

    mix = Sum(inputs=[c1, c2], weights=torch.tensor([[[[w1]], [[w2]]]]))

    # Enumerate k in {0,1,2}
    m = w1 * p1.view(-1) + w2 * p2.view(-1)
    z = float(torch.sum(m.pow(2)).item())

    from spflow.utils.inner_product import log_self_inner_product_scalar

    logZ = log_self_inner_product_scalar(mix)
    torch.testing.assert_close(
        logZ, torch.tensor(math.log(z), dtype=logZ.dtype, device=logZ.device), rtol=1e-6, atol=1e-6
    )


def test_socs_sample_runs_for_monotone_component():
    comp = _make_normal_component(mu=0.0, sigma=1.0)
    model = SOCS([comp])

    cache = Cache()
    cache.extras["socs_mh_steps"] = 5
    cache.extras["socs_mh_burn_in"] = 0
    samples = model.sample(num_samples=13, cache=cache)
    assert samples.shape == (13, 1)
    assert torch.isfinite(samples).all()


def test_socs_sample_signed_component_matches_exact_bernoulli_distribution():
    torch.manual_seed(0)
    p1 = 0.25
    p2 = 0.8
    w1 = 0.9
    w2 = -0.2

    probs1 = torch.tensor([[[p1]]], dtype=torch.get_default_dtype())
    probs2 = torch.tensor([[[p2]]], dtype=torch.get_default_dtype())
    b1 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=probs1)
    b2 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=probs2)
    comp = SignedSum(
        inputs=[b1, b2], out_channels=1, num_repetitions=1, weights=torch.tensor([[[[w1]], [[w2]]]])
    )
    model = SOCS([comp])

    p10, p11 = _bernoulli_probs(p1)
    p20, p21 = _bernoulli_probs(p2)
    c0 = w1 * p10 + w2 * p20
    c1 = w1 * p11 + w2 * p21
    z = c0 * c0 + c1 * c1
    p1_exact = (c1 * c1) / z

    cache = Cache()
    cache.extras["socs_mcmc_steps"] = 30
    cache.extras["socs_mcmc_burn_in"] = 10
    samples = model.sample(num_samples=2_000, cache=cache)
    freq1 = samples.round().clamp(0, 1).mean().item()

    assert abs(freq1 - p1_exact) < 0.05


def test_socs_multi_channel_normal_matches_per_channel_squared_density():
    # One SOCS component with 2 channels should yield two independent squared-Normal densities.
    loc = torch.tensor([[[0.0], [1.0]]], dtype=torch.get_default_dtype())  # (F=1,C=2,R=1)
    scale = torch.tensor([[[1.2], [0.5]]], dtype=torch.get_default_dtype())
    comp = Normal(scope=Scope([0]), out_channels=2, num_repetitions=1, loc=loc, scale=scale)
    model = SOCS([comp])

    x = torch.linspace(-2.0, 2.0, steps=9).unsqueeze(1)
    ll = model.log_likelihood(x).squeeze(-1).squeeze(1)  # (B,C)

    t0 = torch.distributions.Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.2 / math.sqrt(2.0)))
    t1 = torch.distributions.Normal(loc=torch.tensor(1.0), scale=torch.tensor(0.5 / math.sqrt(2.0)))

    expected0 = t0.log_prob(x.squeeze(1))
    expected1 = t1.log_prob(x.squeeze(1))
    expected = torch.stack([expected0, expected1], dim=1)

    torch.testing.assert_close(ll, expected.to(dtype=ll.dtype, device=ll.device), rtol=1e-4, atol=1e-4)


def test_socs_multi_feature_cat_normal_matches_per_feature_squared_density():
    # Two independent features (scopes {0} and {1}) should be normalized per feature.
    n0 = Normal(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[[0.3]]], dtype=torch.get_default_dtype()),
        scale=torch.tensor([[[1.1]]], dtype=torch.get_default_dtype()),
    )
    n1 = Normal(
        scope=Scope([1]),
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[[-0.4]]], dtype=torch.get_default_dtype()),
        scale=torch.tensor([[[0.7]]], dtype=torch.get_default_dtype()),
    )
    from spflow.modules.ops.cat import Cat

    comp = Cat(inputs=[n0, n1], dim=1)
    model = SOCS([comp])

    x = torch.tensor([[0.0, 0.0], [1.0, -1.0], [-0.5, 0.2]], dtype=torch.get_default_dtype())
    ll = model.log_likelihood(x).squeeze(-1).squeeze(-1)  # (B,F)

    t0 = torch.distributions.Normal(loc=torch.tensor(0.3), scale=torch.tensor(1.1 / math.sqrt(2.0)))
    t1 = torch.distributions.Normal(loc=torch.tensor(-0.4), scale=torch.tensor(0.7 / math.sqrt(2.0)))
    expected = torch.stack([t0.log_prob(x[:, 0]), t1.log_prob(x[:, 1])], dim=1)

    torch.testing.assert_close(ll, expected.to(dtype=ll.dtype, device=ll.device), rtol=1e-4, atol=1e-4)


def test_socs_log_partition_cache_is_per_module():
    # Regression test: SOCS must not reuse another SOCS' log-partition when sharing a Cache.
    sigma1 = 0.9
    sigma2 = 1.7

    x = torch.zeros(3, 1, dtype=torch.get_default_dtype())
    cache = Cache()

    m1 = SOCS([_make_normal_component(mu=0.0, sigma=sigma1)])
    _ = m1.log_likelihood(x, cache=cache)
    logZ1 = cache.extras["socs_logZ"][0, 0, 0].detach().cpu()
    expected1 = logZ1.new_tensor(math.log(1.0 / (2.0 * math.sqrt(math.pi) * sigma1)))
    torch.testing.assert_close(logZ1, expected1, rtol=0.0, atol=1e-6)

    m2 = SOCS([_make_normal_component(mu=0.0, sigma=sigma2)])
    _ = m2.log_likelihood(x, cache=cache)
    logZ2 = cache.extras["socs_logZ"][0, 0, 0].detach().cpu()
    expected2 = logZ2.new_tensor(math.log(1.0 / (2.0 * math.sqrt(math.pi) * sigma2)))
    torch.testing.assert_close(logZ2, expected2, rtol=0.0, atol=1e-6)

    assert not torch.allclose(logZ1, logZ2)


class _MargNoneModule(Module):
    def __init__(self, scope: Scope, out_channels: int = 1) -> None:
        super().__init__()
        self.scope = scope
        self.in_shape = ModuleShape(features=len(scope.query), channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=1, channels=out_channels, repetitions=1)
        self._feature_to_scope = torch.tensor([[0]], dtype=torch.int64).numpy()

    @property
    def feature_to_scope(self):
        return self._feature_to_scope

    def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:
        return torch.zeros(
            (data.shape[0], 1, self.out_shape.channels, 1), dtype=data.dtype, device=data.device
        )

    def sample(
        self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None
    ) -> torch.Tensor:
        n = 1 if num_samples is None else num_samples
        return torch.zeros((n, len(self.scope.query)))

    def _sample(
        self,
        data: torch.Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> torch.Tensor:
        del sampling_ctx
        del cache
        return torch.zeros((data.shape[0], len(self.scope.query)), dtype=data.dtype, device=data.device)

    def expectation_maximization(self, data, bias_correction=True, cache=None) -> None:
        return None

    def maximum_likelihood_estimation(self, data, weights=None, cache=None) -> None:
        return None

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return None


def test_socs_constructor_validation_and_feature_scope():
    with pytest.raises(ValueError):
        SOCS([])

    c0 = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    c1 = Normal(scope=Scope([1]), out_channels=1, num_repetitions=1)
    with pytest.raises(ShapeError):
        SOCS([c0, c1])

    c2 = Normal(scope=Scope([0]), out_channels=2, num_repetitions=1)
    with pytest.raises(ShapeError):
        SOCS([c0, c2])

    model = SOCS([c0])
    assert model.feature_to_scope.shape == c0.feature_to_scope.shape


def test_socs_signed_eval_sum_product_and_cache_hit():
    x0 = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    x1 = Normal(scope=Scope([1]), out_channels=1, num_repetitions=1)
    mix = Sum(inputs=[x0, x0], out_channels=1, num_repetitions=1)
    prod = Product([x0, x1])

    data = torch.zeros((3, 2))
    cache = Cache()

    sum_logabs, sum_sign = _signed_eval(mix, data[:, :1], cache)
    assert sum_logabs.shape == (3, 1, 1, 1)
    assert torch.equal(sum_sign, torch.ones_like(sum_sign, dtype=torch.int8))

    cached_logabs, cached_sign = _signed_eval(mix, data[:, :1], cache)
    assert cached_logabs is sum_logabs
    assert cached_sign is sum_sign

    prod_logabs, prod_sign = _signed_eval(prod, data, Cache())
    assert prod_logabs.shape == (3, 1, 1, 1)
    assert prod_sign.shape == (3, 1, 1, 1)


def test_socs_log_partition_cache_hit_and_unsupported_methods():
    comp = _make_normal_component(mu=0.0, sigma=1.0)
    model = SOCS([comp])
    x = torch.zeros((2, 1))
    cache = Cache()
    _ = model.log_likelihood(x, cache=cache)
    first = cache.extras["socs_logZ"]
    _ = model.log_likelihood(x, cache=cache)
    second = cache.extras["socs_logZ"]
    assert first is second

    with pytest.raises(UnsupportedOperationError):
        model._expectation_maximization_step(x, cache=Cache())
    with pytest.raises(AttributeError):
        model.maximum_likelihood_estimation(x)


def test_socs_marginalize_none_and_prune_paths():
    model_none = SOCS([_MargNoneModule(scope=Scope([0]))])
    assert model_none.marginalize([0], prune=True) is None

    comp = Product(
        [
            Normal(scope=Scope([0]), out_channels=1, num_repetitions=1),
            Normal(scope=Scope([1]), out_channels=1, num_repetitions=1),
        ]
    )
    model = SOCS([comp])
    out_prune = model.marginalize([0], prune=True)
    out_no_prune = model.marginalize([0], prune=False)
    assert isinstance(out_prune, SOCS)
    assert isinstance(out_no_prune, SOCS)


def test_socs_sample_error_paths_and_default_num_samples():
    scalar = _make_normal_component(mu=0.0, sigma=1.0)
    model = SOCS([scalar])

    with pytest.raises(UnsupportedOperationError):
        model.sample(is_mpe=True)

    with pytest.raises(UnsupportedOperationError):
        model.sample(data=torch.zeros((2, 1)))

    non_scalar = SOCS([Normal(scope=Scope([0]), out_channels=2, num_repetitions=1)])
    with pytest.raises(UnsupportedOperationError):
        non_scalar.sample(num_samples=1)

    cache_bad_steps = Cache()
    cache_bad_steps.extras["socs_mcmc_steps"] = 0
    with pytest.raises(ValueError):
        model.sample(num_samples=1, cache=cache_bad_steps)

    cache_bad_burn = Cache()
    cache_bad_burn.extras["socs_mcmc_burn_in"] = -1
    with pytest.raises(ValueError):
        model.sample(num_samples=1, cache=cache_bad_burn)

    out = model.sample()
    assert out.shape == (1, 1)

    nan_data = torch.full((4, 1), float("nan"))
    out_data = model.sample(data=nan_data)
    assert out_data.shape == (4, 1)


def test_socs_sample_component_loop_continue_branch():
    comp0 = _make_normal_component(mu=0.0, sigma=1.0)
    comp1 = _make_normal_component(mu=0.5, sigma=1.1)
    model = SOCS([comp0, comp1])
    cache = Cache()
    cache.extras["socs_mcmc_steps"] = 1
    cache.extras["socs_mcmc_burn_in"] = 0
    samples = model.sample(num_samples=1, cache=cache)
    assert samples.shape == (1, 1)
