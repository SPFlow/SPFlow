import torch

from spflow.utils.diff_sampling import (
    DiffSampleMethod,
    sample_categorical_differentiably,
    select_with_soft_or_hard,
    simple_st_one_hot,
)


def test_simple_st_one_hot_shape_and_normalization():
    logits = torch.randn(4, 3, 7)
    y = simple_st_one_hot(logits=logits, dim=-1, is_mpe=False)
    assert y.shape == logits.shape
    torch.testing.assert_close(y.sum(dim=-1), torch.ones_like(y.sum(dim=-1)))


def test_simple_st_one_hot_backward_has_grads():
    logits = torch.randn(8, 5, requires_grad=True)
    y = simple_st_one_hot(logits=logits, dim=-1, is_mpe=False)
    loss = (y * torch.linspace(0.0, 1.0, steps=5, device=y.device)).sum()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_sample_categorical_differentiably_simple():
    logits = torch.randn(2, 6, 4)
    s = sample_categorical_differentiably(dim=-1, is_mpe=False, logits=logits, method=DiffSampleMethod.SIMPLE)
    assert s.shape == logits.shape
    torch.testing.assert_close(s.sum(dim=-1), torch.ones_like(s.sum(dim=-1)))


def test_sample_categorical_differentiably_gumbel():
    logits = torch.randn(2, 3, 5)
    s = sample_categorical_differentiably(
        dim=-1,
        is_mpe=False,
        hard=False,
        tau=0.7,
        logits=logits,
        method=DiffSampleMethod.GUMBEL,
    )
    assert s.shape == logits.shape
    torch.testing.assert_close(s.sum(dim=-1), torch.ones_like(s.sum(dim=-1)), rtol=1e-4, atol=1e-5)


def test_select_with_soft_or_hard_supports_soft_and_hard_paths():
    x = torch.tensor([[[1.0, 3.0, 5.0]]])

    hard = select_with_soft_or_hard(x, index=torch.tensor([[[2]]]), dim=2)
    torch.testing.assert_close(hard, torch.tensor([[5.0]]))

    selector = torch.tensor([[[0.2, 0.3, 0.5]]])
    soft = select_with_soft_or_hard(x, selector=selector, dim=2)
    torch.testing.assert_close(soft, torch.tensor([[3.6]]))
