import torch

from spflow.meta import Scope
from spflow.modules.leaves.geometric import GeometricWithDifferentiableSamplingSIMPLE
from spflow.modules.leaves.histogram import Histogram
from spflow.modules.leaves.hypergeometric import Hypergeometric
from spflow.modules.leaves.negative_binomial import NegativeBinomialWithDifferentiableSamplingSIMPLE
from spflow.modules.leaves.poisson import PoissonWithDifferentiableSamplingSIMPLE


def test_geometric_simple_rsample_shapes_bounds_and_grads():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 1, requires_grad=True)
    dist = GeometricWithDifferentiableSamplingSIMPLE(logits=logits, validate_args=True)
    samples = dist.rsample((5,))

    assert samples.shape == (5, 2, 3, 1)
    assert torch.isfinite(samples).all()
    assert ((samples - samples.round()).abs() == 0).all()
    assert (samples >= 0).all()

    samples.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_poisson_simple_rsample_shapes_bounds_and_grads():
    torch.manual_seed(0)
    log_rate = torch.randn(2, 3, 1, requires_grad=True)
    rate = torch.exp(log_rate)
    dist = PoissonWithDifferentiableSamplingSIMPLE(rate=rate, validate_args=True)
    samples = dist.rsample((6,))

    assert samples.shape == (6, 2, 3, 1)
    assert torch.isfinite(samples).all()
    assert ((samples - samples.round()).abs() == 0).all()
    assert (samples >= 0).all()

    samples.mean().backward()
    assert log_rate.grad is not None
    assert torch.isfinite(log_rate.grad).all()


def test_negative_binomial_simple_rsample_shapes_bounds_and_grads():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 1, requires_grad=True)
    total_count = torch.full((2, 3, 1), 4.0)
    dist = NegativeBinomialWithDifferentiableSamplingSIMPLE(
        total_count=total_count, logits=logits, validate_args=True
    )
    samples = dist.rsample((4,))

    assert samples.shape == (4, 2, 3, 1)
    assert torch.isfinite(samples).all()
    assert ((samples - samples.round()).abs() == 0).all()
    assert (samples >= 0).all()

    samples.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_hypergeometric_simple_rsample_is_in_support_and_finite():
    torch.manual_seed(0)
    leaf = Hypergeometric(
        scope=Scope([0]),
        K=torch.tensor([[[2.0]]]),
        N=torch.tensor([[[5.0]]]),
        n=torch.tensor([[[2.0]]]),
    )
    dist = leaf.distribution(with_differentiable_sampling=True)
    samples = dist.rsample((10,))

    assert samples.shape == (10, 1, 1, 1)
    assert torch.isfinite(samples).all()
    assert ((samples - samples.round()).abs() == 0).all()
    assert (samples >= 0).all()
    assert (samples <= 2).all()


def test_histogram_simple_rsample_shapes_support_and_grads():
    torch.manual_seed(0)
    leaf = Histogram(scope=Scope([0]), bin_edges=torch.tensor([0.0, 1.0, 3.0]), out_channels=2, num_repetitions=1)
    dist = leaf.distribution(with_differentiable_sampling=True)
    samples = dist.rsample((7,))

    assert samples.shape == (7, 1, 2, 1)
    assert torch.isfinite(samples).all()
    assert (samples >= 0.0).all()
    assert (samples < 3.0).all()

    samples.mean().backward()
    assert leaf.logits.grad is not None
    assert torch.isfinite(leaf.logits.grad).all()
