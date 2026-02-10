"""Lightweight tensor tracing helpers for APC debugging."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

_TRACE_ENABLED = False
_TRACE_PREFIX = "SPFLOW"
_TRACE_MAX_EVENTS = 400
_TRACE_MAX_VALUES = 6
_TRACE_EVENT_COUNT = 0


def configure_trace(
    *,
    enabled: bool,
    prefix: str = "SPFLOW",
    max_events: int = 400,
    max_values: int = 6,
) -> None:
    """Configure runtime tracing behavior."""
    global _TRACE_ENABLED, _TRACE_PREFIX, _TRACE_MAX_EVENTS, _TRACE_MAX_VALUES, _TRACE_EVENT_COUNT
    _TRACE_ENABLED = enabled
    _TRACE_PREFIX = prefix
    _TRACE_MAX_EVENTS = max(1, int(max_events))
    _TRACE_MAX_VALUES = max(1, int(max_values))
    _TRACE_EVENT_COUNT = 0


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw.lower() in {"1", "true", "yes", "on"}


def _trace_active() -> bool:
    if _TRACE_ENABLED:
        return True
    return _env_truthy("APC_TRACE")


def _consume_event() -> bool:
    global _TRACE_EVENT_COUNT
    if not _trace_active():
        return False
    if _TRACE_EVENT_COUNT >= _TRACE_MAX_EVENTS:
        return False
    _TRACE_EVENT_COUNT += 1
    return True


def _format_values(x: Tensor, max_values: int) -> str:
    if x.numel() == 0:
        return "[]"
    flat = x.detach().reshape(-1)
    if flat.is_floating_point():
        finite = flat[torch.isfinite(flat)]
        flat = finite if finite.numel() > 0 else flat
    vals = flat[: max_values].tolist()
    return str([float(v) for v in vals])


def trace_tensor(name: str, value: Tensor | None) -> None:
    """Print shape/statistics for a tensor."""
    if not _consume_event():
        return
    if value is None:
        print(f"[{_TRACE_PREFIX}][TRACE] {name}: <None>", flush=True)
        return

    x = value.detach()
    numel = int(x.numel())
    nan_count = int(torch.isnan(x).sum().item()) if x.is_floating_point() else 0
    inf_count = int(torch.isinf(x).sum().item()) if x.is_floating_point() else 0
    finite_count = int(torch.isfinite(x).sum().item()) if x.is_floating_point() else numel

    stats = ""
    if numel > 0:
        if x.is_floating_point():
            finite = x[torch.isfinite(x)]
            if finite.numel() > 0:
                stats = (
                    f" min={finite.min().item():.6g}"
                    f" max={finite.max().item():.6g}"
                    f" mean={finite.mean().item():.6g}"
                )
            else:
                stats = " min=nan max=nan mean=nan"
        else:
            stats = f" min={x.min().item()} max={x.max().item()} mean={x.float().mean().item():.6g}"

    print(
        f"[{_TRACE_PREFIX}][TRACE] {name}:"
        f" shape={tuple(x.shape)}"
        f" dtype={x.dtype}"
        f" device={x.device}"
        f" requires_grad={bool(value.requires_grad)}"
        f" finite={finite_count}/{numel}"
        f" nan={nan_count}"
        f" inf={inf_count}"
        f"{stats}"
        f" values={_format_values(x, _TRACE_MAX_VALUES)}",
        flush=True,
    )


def trace_value(name: str, value: Any) -> None:
    """Print a scalar/string debug value."""
    if not _consume_event():
        return
    print(f"[{_TRACE_PREFIX}][TRACE] {name}: {value}", flush=True)


def trace_sampling_context(name: str, ctx: Any) -> None:
    """Print sampling-context routing tensors if present."""
    if not _trace_active():
        return
    if ctx is None:
        trace_value(name, "<None>")
        return
    trace_tensor(f"{name}.channel_index", getattr(ctx, "channel_index", None))
    trace_tensor(f"{name}.mask", getattr(ctx, "mask", None))
    trace_tensor(f"{name}.repetition_idx", getattr(ctx, "repetition_idx", None))
    trace_tensor(f"{name}.channel_select", getattr(ctx, "channel_select", None))
    trace_tensor(f"{name}.repetition_select", getattr(ctx, "repetition_select", None))
