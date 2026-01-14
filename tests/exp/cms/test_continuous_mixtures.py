import torch

from spflow.exp.cms import (
    JointLogLikelihood,
    LatentOptimizationConfig,
    learn_continuous_mixture_cltree,
    learn_continuous_mixture_factorized,
)
from spflow.meta import Scope
from spflow.modules.leaves import CLTree


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
