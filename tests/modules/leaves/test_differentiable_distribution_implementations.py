import torch

from spflow.modules.leaves.binomial import (
    BinomialWithDifferentiableSamplingNormal,
    BinomialWithDifferentiableSamplingSIMPLE,
)
from spflow.modules.leaves.categorical import CategoricalWithDifferentiableSampling
from spflow.modules.leaves.geometric import GeometricWithDifferentiableSamplingSIMPLE
from spflow.modules.leaves.negative_binomial import NegativeBinomialWithDifferentiableSamplingSIMPLE
from spflow.modules.leaves.poisson import PoissonWithDifferentiableSamplingSIMPLE


def test_binomial_normal_rsample_shapes_bounds_and_grads():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 1, requires_grad=True)
    total_count = torch.full((2, 3, 1), 10.0)

    dist = BinomialWithDifferentiableSamplingNormal(
        total_count=total_count, logits=logits, validate_args=True
    )
    samples = dist.rsample((7,))

    assert samples.shape == (7, 2, 3, 1)
    assert torch.isfinite(samples).all()
    assert ((samples - samples.round()).abs() == 0).all()
    assert (samples >= 0.0).all()
    assert (samples <= total_count).all()

    loss = samples.mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_binomial_simple_rsample_shapes_bounds_and_grads_with_varying_total_count():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 1, requires_grad=True)
    total_count = torch.tensor([[[2.0], [4.0], [6.0]], [[1.0], [3.0], [5.0]]])

    dist = BinomialWithDifferentiableSamplingSIMPLE(
        total_count=total_count, logits=logits, validate_args=True
    )
    samples = dist.rsample((5,))

    assert samples.shape == (5, 2, 3, 1)
    assert torch.isfinite(samples).all()
    assert ((samples - samples.round()).abs() == 0).all()
    assert (samples >= 0.0).all()

    total_count_expanded = total_count.unsqueeze(0).expand(samples.shape)
    assert (samples <= total_count_expanded).all()

    loss = samples.float().mean() + logits.mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_categorical_simple_rsample_shapes_bounds_and_grads():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 1, 5, requires_grad=True)
    dist = CategoricalWithDifferentiableSampling(logits=logits, validate_args=True)

    samples = dist.rsample((7,))

    assert samples.shape == (7, 2, 3, 1)
    assert torch.isfinite(samples).all()
    assert ((samples - samples.round()).abs() == 0).all()
    assert (samples >= 0).all()
    assert (samples <= 4).all()

    samples.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


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
