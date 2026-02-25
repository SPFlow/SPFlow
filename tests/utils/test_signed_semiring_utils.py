"""Tests for spflow.utils.signed_semiring."""

import torch

from spflow.utils.signed_semiring import logabs_of, sign_of, signed_logsumexp


def test_sign_of_returns_int8_signs() -> None:
    x = torch.tensor([-2.0, 0.0, 4.0])
    out = sign_of(x)
    assert out.dtype == torch.int8
    torch.testing.assert_close(out, torch.tensor([-1, 0, 1], dtype=torch.int8), rtol=0.0, atol=0.0)


def test_logabs_of_with_and_without_eps() -> None:
    x = torch.tensor([0.0, 2.0])
    out_no_eps = logabs_of(x)
    out_eps = logabs_of(x, eps=1e-6)
    # Keep exact zeros at -inf unless eps is requested, so downstream algebra can detect hard zeros.
    assert torch.isneginf(out_no_eps[0])
    assert torch.isfinite(out_eps).all()


def test_signed_logsumexp_keepdim_and_nokeepdim() -> None:
    logabs_terms = torch.log(torch.tensor([[2.0, 3.0], [2.0, 3.0]]))
    sign_terms = torch.tensor([[1, -1], [1, -1]], dtype=torch.int8)

    out_keep, sign_keep = signed_logsumexp(logabs_terms, sign_terms, dim=0, keepdim=True)
    out, sign = signed_logsumexp(logabs_terms, sign_terms, dim=0, keepdim=False)

    assert out_keep.shape == (1, 2)
    assert sign_keep.shape == (1, 2)
    assert out.shape == (2,)
    assert sign.shape == (2,)
    # Cancellation is exact here for each column: (+v) + (+v) and (-v) + (-v)
    torch.testing.assert_close(sign, torch.tensor([1, -1], dtype=torch.int8), rtol=0.0, atol=0.0)


def test_signed_logsumexp_all_neg_inf_branch() -> None:
    logabs_terms = torch.full((2, 3), float("-inf"))
    sign_terms = torch.ones((2, 3), dtype=torch.int8)
    out_logabs, out_sign = signed_logsumexp(logabs_terms, sign_terms, dim=0, keepdim=True, eps=1e-8)
    assert torch.isneginf(out_logabs).all()
    # When total magnitude is exactly zero, sign must be neutral to avoid fake positive/negative mass.
    torch.testing.assert_close(out_sign, torch.zeros_like(out_sign), rtol=0.0, atol=0.0)
