import torch

from spflow.modules.leaves.binomial import (
    BinomialWithDifferentiableSamplingNormal,
    BinomialWithDifferentiableSamplingSIMPLE,
)


def test_binomial_normal_rsample_shapes_bounds_and_grads():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 1, requires_grad=True)
    total_count = torch.full((2, 3, 1), 10.0)

    dist = BinomialWithDifferentiableSamplingNormal(total_count=total_count, logits=logits, validate_args=True)
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

    dist = BinomialWithDifferentiableSamplingSIMPLE(total_count=total_count, logits=logits, validate_args=True)
    samples = dist.rsample((5,))

    assert samples.shape == (5, 2, 3, 1)
    assert torch.isfinite(samples).all()
    assert ((samples - samples.round()).abs() == 0).all()
    assert (samples >= 0.0).all()

    total_count_expanded = total_count.unsqueeze(0).expand(samples.shape)
    assert (samples <= total_count_expanded).all()

    loss = (samples.float().mean() + logits.mean())
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
