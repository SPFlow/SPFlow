import torch

from spflow.modules.leaves.categorical import CategoricalWithDifferentiableSampling


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

