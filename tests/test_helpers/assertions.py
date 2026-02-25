"""Shared assertion helpers for contract tests."""

from __future__ import annotations

import torch


def assert_finite_tensor(tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all()


def assert_probabilities(proba: torch.Tensor, *, atol: float = 1e-5) -> None:
    assert torch.allclose(proba.sum(dim=-1), torch.ones(proba.shape[0], device=proba.device), atol=atol)
