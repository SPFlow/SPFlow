import pytest
import torch

from spflow.utils.signed_semiring import logabs_of
from spflow.utils.signed_semiring import signed_logsumexp


def test_logabs_of_with_and_without_eps():
    x = torch.tensor([-2.0, 0.0, 3.0])
    out_no_eps = logabs_of(x)
    out_eps = logabs_of(x, eps=1e-6)
    assert torch.isneginf(out_no_eps[1])
    assert torch.isfinite(out_eps).all()


def test_signed_logsumexp_empty_input_raises():
    with pytest.raises(ValueError):
        signed_logsumexp(
            logabs_terms=torch.empty((0, 1)),
            sign_terms=torch.empty((0, 1), dtype=torch.int8),
            dim=0,
        )


def test_signed_logsumexp_all_neg_inf_and_eps_branches():
    logabs_terms = torch.tensor([[-torch.inf, -torch.inf]])
    sign_terms = torch.tensor([[1, -1]], dtype=torch.int8)
    out_logabs, out_sign = signed_logsumexp(logabs_terms, sign_terms, dim=0, keepdim=True, eps=1e-8)
    assert torch.isneginf(out_logabs).all()
    assert torch.equal(out_sign, torch.zeros_like(out_sign))
