import pytest
import torch

from spflow.exceptions import UnsupportedOperationError
from spflow.meta import Scope
from spflow.modules.leaves import (
    Bernoulli,
    Binomial,
    Categorical,
    CLTree,
    Exponential,
    Gamma,
    Geometric,
    Histogram,
    Hypergeometric,
    Laplace,
    LogNormal,
    NegativeBinomial,
    Normal,
    PiecewiseLinear,
    Poisson,
    Uniform,
)


@pytest.mark.parametrize(
    "leaf_ctor",
    [
        lambda: Normal(scope=Scope([0])),
        lambda: Laplace(scope=Scope([0])),
        lambda: LogNormal(scope=Scope([0])),
        lambda: Gamma(scope=Scope([0])),
        lambda: Exponential(scope=Scope([0])),
        lambda: Uniform(
            scope=Scope([0]),
            low=torch.tensor([[[0.0]]]),
            high=torch.tensor([[[1.0]]]),
        ),
        lambda: Bernoulli(scope=Scope([0])),
        lambda: Binomial(scope=Scope([0]), total_count=torch.tensor([[[4.0]]])),
        lambda: Categorical(scope=Scope([0]), K=3),
        lambda: Geometric(scope=Scope([0])),
        lambda: Poisson(scope=Scope([0])),
        lambda: NegativeBinomial(scope=Scope([0]), total_count=torch.tensor([[[4.0]]])),
        lambda: Hypergeometric(
            scope=Scope([0]),
            K=torch.tensor([[[2.0]]]),
            N=torch.tensor([[[5.0]]]),
            n=torch.tensor([[[2.0]]]),
        ),
        lambda: Histogram(scope=Scope([0]), bin_edges=torch.tensor([0.0, 1.0, 2.0])),
    ],
)
def test_supported_leaves_expose_differentiable_distribution(leaf_ctor):
    leaf = leaf_ctor()

    dist = leaf.distribution(with_differentiable_sampling=True)
    samples = dist.rsample((3,))

    assert hasattr(dist, "rsample")
    assert samples.shape[0] == 3
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "leaf_ctor",
    [
        lambda: PiecewiseLinear(scope=Scope([0])),
        lambda: CLTree(scope=Scope([0, 1]), K=2),
    ],
)
def test_unsupported_leaves_raise_for_differentiable_distribution(leaf_ctor):
    leaf = leaf_ctor()

    with pytest.raises((NotImplementedError, UnsupportedOperationError)):
        leaf.distribution(with_differentiable_sampling=True)
