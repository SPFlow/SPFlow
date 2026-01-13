import math

import torch

from spflow.meta.data.scope import Scope
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.leaves.exponential import Exponential
from spflow.modules.leaves.gamma import Gamma
from spflow.modules.leaves.laplace import Laplace
from spflow.modules.leaves.log_normal import LogNormal
from spflow.modules.leaves.poisson import Poisson
from spflow.utils.inner_product import inner_product_matrix


def test_exponential_inner_product_closed_form():
    r1 = torch.tensor([[[0.7]]])
    r2 = torch.tensor([[[2.3]]])
    a = Exponential(scope=Scope([0]), out_channels=1, num_repetitions=1, rate=r1)
    b = Exponential(scope=Scope([0]), out_channels=1, num_repetitions=1, rate=r2)
    k = inner_product_matrix(a, b)
    expected = (0.7 * 2.3) / (0.7 + 2.3)
    torch.testing.assert_close(k[0, 0, 0, 0], torch.tensor(expected, dtype=k.dtype, device=k.device))


def test_laplace_inner_product_matches_known_special_cases():
    # Same scale, different loc: ∫ Lap(a,b)^2 = exp(-d/b)*(b+d)/(4 b^2)
    b = 1.25
    d = 0.8
    loc1 = torch.tensor([[[0.0]]])
    loc2 = torch.tensor([[[d]]])
    scale = torch.tensor([[[b]]])
    a = Laplace(scope=Scope([0]), out_channels=1, num_repetitions=1, loc=loc1, scale=scale)
    c = Laplace(scope=Scope([0]), out_channels=1, num_repetitions=1, loc=loc2, scale=scale)
    k = inner_product_matrix(a, c)
    expected = math.exp(-d / b) * (b + d) / (4.0 * b * b)
    torch.testing.assert_close(
        k[0, 0, 0, 0], torch.tensor(expected, dtype=k.dtype, device=k.device), rtol=1e-6, atol=1e-6
    )

    # Different scale, same loc: ∫ Lap(0,b1)*Lap(0,b2) = 1/(2(b1+b2))
    b1 = 0.7
    b2 = 1.9
    a = Laplace(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[[0.0]]]),
        scale=torch.tensor([[[b1]]]),
    )
    c = Laplace(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[[0.0]]]),
        scale=torch.tensor([[[b2]]]),
    )
    k = inner_product_matrix(a, c)
    expected = 1.0 / (2.0 * (b1 + b2))
    torch.testing.assert_close(
        k[0, 0, 0, 0], torch.tensor(expected, dtype=k.dtype, device=k.device), rtol=1e-6, atol=1e-6
    )


def test_lognormal_inner_product_matches_monte_carlo_estimate():
    torch.manual_seed(0)
    mu1, s1 = 0.2, 0.6
    mu2, s2 = -0.4, 0.9

    a = LogNormal(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[[mu1]]]),
        scale=torch.tensor([[[s1]]]),
    )
    b = LogNormal(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[[mu2]]]),
        scale=torch.tensor([[[s2]]]),
    )

    # Estimate ∫ f1(x) f2(x) dx = E_{X~f1}[f2(X)] via Monte Carlo.
    dist1 = torch.distributions.LogNormal(loc=torch.tensor(mu1), scale=torch.tensor(s1))
    dist2 = torch.distributions.LogNormal(loc=torch.tensor(mu2), scale=torch.tensor(s2))
    x = dist1.sample((50_000,))
    mc = torch.exp(dist2.log_prob(x)).mean()

    k = inner_product_matrix(a, b)[0, 0, 0, 0]
    torch.testing.assert_close(k, mc.to(dtype=k.dtype, device=k.device), rtol=5e-3, atol=5e-4)


def test_poisson_inner_product_matches_bessel_series():
    lam1 = 0.8
    lam2 = 1.7
    a = Poisson(scope=Scope([0]), out_channels=1, num_repetitions=1, rate=torch.tensor([[[lam1]]]))
    b = Poisson(scope=Scope([0]), out_channels=1, num_repetitions=1, rate=torch.tensor([[[lam2]]]))

    k = inner_product_matrix(a, b)[0, 0, 0, 0]
    i0 = getattr(torch, "i0", torch.special.i0)
    expected = math.exp(-(lam1 + lam2)) * i0(torch.tensor(2.0 * math.sqrt(lam1 * lam2))).item()
    torch.testing.assert_close(
        k, torch.tensor(expected, dtype=k.dtype, device=k.device), rtol=1e-7, atol=1e-10
    )


def test_gamma_inner_product_matches_monte_carlo_estimate():
    torch.manual_seed(0)
    a1, b1 = 2.4, 1.3
    a2, b2 = 1.6, 0.9
    a = Gamma(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        concentration=torch.tensor([[[a1]]]),
        rate=torch.tensor([[[b1]]]),
    )
    b = Gamma(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        concentration=torch.tensor([[[a2]]]),
        rate=torch.tensor([[[b2]]]),
    )

    # Estimate ∫ f1(x) f2(x) dx = E_{X~f1}[f2(X)] via Monte Carlo.
    dist1 = torch.distributions.Gamma(concentration=torch.tensor(a1), rate=torch.tensor(b1))
    dist2 = torch.distributions.Gamma(concentration=torch.tensor(a2), rate=torch.tensor(b2))
    x = dist1.sample((80_000,))
    mc = torch.exp(dist2.log_prob(x)).mean()

    k = inner_product_matrix(a, b)[0, 0, 0, 0]
    torch.testing.assert_close(k, mc.to(dtype=k.dtype, device=k.device), rtol=7e-3, atol=5e-4)


def _make_cltree_log_cpt(
    *,
    parents: list[int],
    K: int,
    root_probs: torch.Tensor,  # (C,K)
    cond_probs: dict[int, torch.Tensor],  # i -> (C,K(child),K(parent))
) -> torch.Tensor:
    C = int(root_probs.shape[0])
    F = len(parents)
    log_cpt = torch.empty((F, C, 1, K, K), dtype=torch.get_default_dtype())
    root = parents.index(-1)

    for i in range(F):
        if i == root:
            rp = root_probs.clamp_min(1e-12)
            logp = rp.log()  # (C,K)
            log_cpt[i, :, 0, :, :] = logp.unsqueeze(-1).expand(-1, -1, K)
        else:
            probs = cond_probs[i].clamp_min(1e-12)  # (C,K,K)
            log_cpt[i, :, 0, :, :] = probs.log()

    return log_cpt


def test_cltree_inner_product_matches_exact_enumeration():
    # Scope of length 3, binary, fixed structure: 0 is root, 1->0, 2->1
    scope = Scope([0, 1, 2])
    parents = [-1, 0, 1]
    K = 2

    # Two-channel trees to exercise (Ca,Cb) pairs.
    root_a = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    root_b = torch.tensor([[0.2, 0.8], [0.5, 0.5]])

    cond_a = {
        1: torch.tensor(
            [
                [[0.9, 0.1], [0.2, 0.8]],  # x1 | x0  (xi rows, xp cols)
                [[0.7, 0.3], [0.4, 0.6]],
            ]
        ),
        2: torch.tensor(
            [
                [[0.6, 0.4], [0.3, 0.7]],  # x2 | x1
                [[0.8, 0.2], [0.1, 0.9]],
            ]
        ),
    }
    cond_b = {
        1: torch.tensor(
            [
                [[0.5, 0.5], [0.1, 0.9]],
                [[0.9, 0.1], [0.3, 0.7]],
            ]
        ),
        2: torch.tensor(
            [
                [[0.4, 0.6], [0.2, 0.8]],
                [[0.7, 0.3], [0.6, 0.4]],
            ]
        ),
    }

    log_cpt_a = _make_cltree_log_cpt(parents=parents, K=K, root_probs=root_a, cond_probs=cond_a)
    log_cpt_b = _make_cltree_log_cpt(parents=parents, K=K, root_probs=root_b, cond_probs=cond_b)

    a = CLTree(
        scope=scope, out_channels=2, num_repetitions=1, K=K, parents=torch.tensor(parents), log_cpt=log_cpt_a
    )
    b = CLTree(
        scope=scope, out_channels=2, num_repetitions=1, K=K, parents=torch.tensor(parents), log_cpt=log_cpt_b
    )

    k = inner_product_matrix(a, b)  # (F, Ca, Cb, R)
    assert k.shape == (3, 2, 2, 1)

    # Brute force over all assignments x0,x1,x2 in {0,1}.
    def p_tree(root, cond1, cond2, x0, x1, x2):
        return root[x0] * cond1[x1, x0] * cond2[x2, x1]

    for ca in range(2):
        for cb in range(2):
            tot = 0.0
            for x0 in range(2):
                for x1 in range(2):
                    for x2 in range(2):
                        pa = p_tree(root_a[ca], cond_a[1][ca], cond_a[2][ca], x0, x1, x2)
                        pb = p_tree(root_b[cb], cond_b[1][cb], cond_b[2][cb], x0, x1, x2)
                        tot += float((pa * pb).item())
            torch.testing.assert_close(
                k[0, ca, cb, 0],
                torch.tensor(tot, dtype=k.dtype, device=k.device),
                rtol=1e-6,
                atol=1e-6,
            )

    # Features 1.. must be 1 so product aggregation stays correct.
    torch.testing.assert_close(k[1:, :, :, :], torch.ones_like(k[1:, :, :, :]))
