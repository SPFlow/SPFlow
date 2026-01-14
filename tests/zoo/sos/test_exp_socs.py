import math

import numpy as np
import torch
from scipy import integrate

from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.normal import Normal
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache
from spflow.zoo.sos import ExpSOCS
from spflow.zoo.sos import SignedSum
from spflow.zoo.sos.socs import _signed_eval


def test_exp_socs_discrete_bernoulli_matches_exact_enumeration():
    # One-variable case: p(x) ∝ m(x) * c(x)^2, x in {0,1}.
    p1 = 0.2
    p2 = 0.8
    v1 = 0.35
    v2 = 0.65
    w1 = 1.0
    w2 = -0.5

    b1 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=torch.tensor([[[p1]]]))
    b2 = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=torch.tensor([[[p2]]]))
    mono = Sum(inputs=[b1, b2], out_channels=1, num_repetitions=1, weights=torch.tensor([[[[v1]], [[v2]]]]))
    comp = SignedSum(
        inputs=[b1, b2], out_channels=1, num_repetitions=1, weights=torch.tensor([[[[w1]], [[w2]]]])
    )

    model = ExpSOCS(monotone=mono, components=[comp])

    m0 = v1 * (1.0 - p1) + v2 * (1.0 - p2)
    m1 = v1 * p1 + v2 * p2
    c0 = w1 * (1.0 - p1) + w2 * (1.0 - p2)
    c1 = w1 * p1 + w2 * p2
    z = m0 * c0 * c0 + m1 * c1 * c1

    x = torch.tensor([[0.0], [1.0]], dtype=torch.get_default_dtype())
    cache = Cache()
    ll = model.log_likelihood(x, cache=cache).squeeze(-1).squeeze(-1).squeeze(1)
    expected = torch.tensor(
        [math.log((m0 * c0 * c0) / z), math.log((m1 * c1 * c1) / z)],
        dtype=ll.dtype,
        device=ll.device,
    )
    torch.testing.assert_close(ll, expected, rtol=1e-6, atol=1e-6)

    logZ = cache.extras["exp_socs_logZ"]
    torch.testing.assert_close(
        logZ, torch.tensor(math.log(z), dtype=logZ.dtype, device=logZ.device), rtol=1e-7, atol=1e-10
    )


def test_exp_socs_continuous_normal_partition_matches_scipy_quad():
    # Compare the exact DP partition vs numerical integration for a 1D Normal-based ExpSOCS.
    mu1, s1 = -0.7, 1.2
    mu2, s2 = 0.8, 0.6
    v1, v2 = 0.4, 0.6
    w1, w2 = 1.0, -0.3

    n1 = Normal(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[[mu1]]], dtype=torch.get_default_dtype()),
        scale=torch.tensor([[[s1]]], dtype=torch.get_default_dtype()),
    )
    n2 = Normal(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[[mu2]]], dtype=torch.get_default_dtype()),
        scale=torch.tensor([[[s2]]], dtype=torch.get_default_dtype()),
    )
    mono = Sum(inputs=[n1, n2], out_channels=1, num_repetitions=1, weights=torch.tensor([[[[v1]], [[v2]]]]))
    comp = SignedSum(
        inputs=[n1, n2], out_channels=1, num_repetitions=1, weights=torch.tensor([[[[w1]], [[w2]]]])
    )
    model = ExpSOCS(monotone=mono, components=[comp])

    cache = Cache()
    _ = model.log_likelihood(torch.zeros(1, 1, dtype=torch.get_default_dtype()), cache=cache)
    logZ = float(cache.extras["exp_socs_logZ"].detach().cpu().item())

    def numerator(x: float) -> float:
        xx = torch.tensor([[x]], dtype=torch.get_default_dtype())
        with torch.no_grad():
            log_m = mono.log_likelihood(xx).squeeze().item()
            logabs, sign = _signed_eval(comp, xx, Cache())
            c_val = float(sign.item()) * math.exp(float(logabs.item()))
            return math.exp(log_m) * (c_val * c_val)

    z_num, _err = integrate.quad(numerator, -np.inf, np.inf, epsabs=1e-8, epsrel=1e-8, limit=200)
    assert abs(math.log(z_num) - logZ) < 2e-4
