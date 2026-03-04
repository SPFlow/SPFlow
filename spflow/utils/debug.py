"""Lightweight tensor and module tracing helpers for debugging."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

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
    """Configure runtime tracing behavior.

    Args:
        enabled: If True, tracing is enabled regardless of env vars.
        prefix: Prefix printed in each trace line.
        max_events: Max number of trace events printed before truncation.
        max_values: Max number of scalar values shown for tensor previews.
    """
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
    return _env_truthy("SPFLOW_TRACE") or _env_truthy("APC_TRACE")


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
    vals = flat[:max_values].tolist()
    return str([float(v) for v in vals])


def _iter_tensors(value: Any, path: str) -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield path, value
        return

    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_tensors(child, f"{path}.{key}")
        return

    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            yield from _iter_tensors(child, f"{path}[{idx}]")


def trace_tensor(name: str, value: Tensor | None) -> None:
    """Print shape and summary statistics for a tensor."""
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
    """Print a scalar or string debug value."""
    if not _consume_event():
        return
    print(f"[{_TRACE_PREFIX}][TRACE] {name}: {value}", flush=True)


def trace_tensor_delta(name: str, before: Tensor | None, after: Tensor | None) -> None:
    """Print compact before/after delta statistics for tensors."""
    if not _consume_event():
        return
    if before is None or after is None:
        has_before = before is not None
        has_after = after is not None
        print(
            f"[{_TRACE_PREFIX}][TRACE] {name}: "
            f"cannot compute delta (before={has_before}, after={has_after})",
            flush=True,
        )
        return
    if before.shape != after.shape:
        before_shape = tuple(before.shape)
        after_shape = tuple(after.shape)
        print(
            f"[{_TRACE_PREFIX}][TRACE] {name}: " f"shape mismatch before={before_shape} after={after_shape}",
            flush=True,
        )
        return

    delta = after.detach().to(dtype=torch.float32) - before.detach().to(dtype=torch.float32)
    finite = delta[torch.isfinite(delta)]
    if finite.numel() == 0:
        print(f"[{_TRACE_PREFIX}][TRACE] {name}: delta has no finite values", flush=True)
        return

    abs_delta = finite.abs()
    print(
        f"[{_TRACE_PREFIX}][TRACE] {name}:"
        f" max_abs={abs_delta.max().item():.6g}"
        f" mean_abs={abs_delta.mean().item():.6g}"
        f" rms={(finite.pow(2).mean().sqrt()).item():.6g}"
        f" values={_format_values(finite, _TRACE_MAX_VALUES)}",
        flush=True,
    )


def trace_tensor_tree(name: str, value: Any, *, max_tensors: int = 8) -> None:
    """Trace tensors contained in nested lists/tuples/dicts."""
    if not _trace_active():
        return

    budget = max(1, int(max_tensors))
    traced = 0
    for path, tensor in _iter_tensors(value, name):
        if traced >= budget:
            trace_value(f"{name}.truncated", f"showing first {budget} tensors")
            return
        trace_tensor(path, tensor)
        traced += 1
    if traced == 0:
        trace_value(name, "<no tensors>")


def trace_module_state(
    name: str,
    module: nn.Module,
    *,
    recurse: bool = True,
    include_buffers: bool = True,
    include_gradients: bool = True,
    max_tensors: int = 8,
) -> None:
    """Trace module-level state, then selected parameter/grad/buffer tensors."""
    if not _trace_active():
        return

    named_params = list(module.named_parameters(recurse=recurse))
    named_buffers = list(module.named_buffers(recurse=recurse)) if include_buffers else []

    param_total = sum(int(p.numel()) for _, p in named_params)
    trainable_total = sum(int(p.numel()) for _, p in named_params if p.requires_grad)
    reference = None
    if named_params:
        reference = named_params[0][1]
    elif named_buffers:
        reference = named_buffers[0][1]
    device = reference.device if reference is not None else "n/a"

    if not _consume_event():
        return
    print(
        f"[{_TRACE_PREFIX}][TRACE] {name}:"
        f" module={module.__class__.__name__}"
        f" training={module.training}"
        f" params={len(named_params)}"
        f" trainable_params={trainable_total}/{param_total}"
        f" buffers={len(named_buffers)}"
        f" device={device}",
        flush=True,
    )

    entries: list[tuple[str, Tensor | None]] = []
    for param_name, param in named_params:
        entries.append((f"{name}.param.{param_name}", param))
        if include_gradients:
            entries.append((f"{name}.grad.{param_name}", param.grad))
    for buffer_name, buffer in named_buffers:
        entries.append((f"{name}.buffer.{buffer_name}", buffer))

    budget = max(1, int(max_tensors))
    for idx, (entry_name, tensor) in enumerate(entries):
        if idx >= budget:
            trace_value(f"{name}.truncated", f"showing first {budget} tensors from module state")
            break
        trace_tensor(entry_name, tensor)


def trace_module_io(name: str, inputs: Any, output: Any, *, max_tensors: int = 6) -> None:
    """Trace input/output tensors for one module invocation."""
    if not _trace_active():
        return
    trace_tensor_tree(f"{name}.inputs", inputs, max_tensors=max_tensors)
    trace_tensor_tree(f"{name}.outputs", output, max_tensors=max_tensors)


def attach_module_trace_hooks(
    module: nn.Module,
    name: str,
    *,
    recurse: bool = False,
    trace_inputs: bool = True,
    trace_outputs: bool = True,
    max_tensors: int = 6,
) -> list[RemovableHandle]:
    """Attach forward hooks that emit input/output tensor traces.

    Args:
        module: Module to instrument.
        name: Prefix used in trace output.
        recurse: If True, attach to all submodules (including root module).
        trace_inputs: If True, trace module inputs.
        trace_outputs: If True, trace module outputs.
        max_tensors: Max number of tensors printed for inputs/outputs per call.

    Returns:
        List of hook handles. Pass to :func:`remove_trace_hooks` for cleanup.
    """
    targets = list(module.named_modules()) if recurse else [("", module)]
    handles: list[RemovableHandle] = []

    for module_path, target in targets:
        label = name if module_path == "" else f"{name}.{module_path}"

        def _hook(
            _: nn.Module,
            hook_inputs: tuple[Any, ...],
            hook_output: Any,
            *,
            hook_label: str = label,
        ) -> None:
            if not _trace_active():
                return
            if trace_inputs:
                trace_tensor_tree(f"{hook_label}.inputs", hook_inputs, max_tensors=max_tensors)
            if trace_outputs:
                trace_tensor_tree(f"{hook_label}.outputs", hook_output, max_tensors=max_tensors)

        handles.append(target.register_forward_hook(_hook))

    return handles


def remove_trace_hooks(handles: list[RemovableHandle]) -> None:
    """Remove previously attached trace hooks."""
    for handle in handles:
        handle.remove()


def trace_sampling_context(name: str, ctx: Any) -> None:
    """Print sampling-context routing tensors if present."""
    if not _trace_active():
        return
    if ctx is None:
        trace_value(name, "<None>")
        return
    trace_tensor(f"{name}.mask", getattr(ctx, "mask", None))
    trace_tensor(f"{name}.channel_index", getattr(ctx, "channel_index", None))
    trace_tensor(f"{name}.repetition_index", getattr(ctx, "repetition_index", None))
