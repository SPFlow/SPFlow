"""Tests for debug trace utility helpers."""

import torch
from torch import nn

from spflow.utils.debug import (
    attach_module_trace_hooks,
    configure_trace,
    remove_trace_hooks,
    trace_module_state,
    trace_tensor_delta,
    trace_tensor_tree,
)


def _capture_lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_trace_tensor_tree_traces_nested_payload(capsys) -> None:
    """Trace nested tensors from dict/list structures."""
    configure_trace(enabled=True, prefix="TEST", max_events=50, max_values=3)
    payload = {
        "x": torch.tensor([1.0, 2.0]),
        "meta": [torch.tensor([[3.0]]), {"y": torch.tensor([-1.0])}],
    }

    trace_tensor_tree("payload", payload, max_tensors=8)
    lines = _capture_lines(capsys)

    assert any("payload.x" in line for line in lines)
    assert any("payload.meta[0]" in line for line in lines)
    assert any("payload.meta[1].y" in line for line in lines)


def test_trace_module_state_includes_parameter_and_gradients(capsys) -> None:
    """Trace module state with parameter and gradient entries."""
    configure_trace(enabled=True, prefix="TEST", max_events=80, max_values=3)
    layer = nn.Linear(3, 2, bias=False)
    layer(torch.ones(4, 3)).sum().backward()

    trace_module_state("linear", layer, include_gradients=True, include_buffers=False, max_tensors=4)
    lines = _capture_lines(capsys)

    assert any("linear: module=Linear" in line for line in lines)
    assert any("linear.param.weight" in line for line in lines)
    assert any("linear.grad.weight" in line for line in lines)


def test_attach_module_trace_hooks_traces_forward_io(capsys) -> None:
    """Trace module inputs/outputs via forward hooks."""
    configure_trace(enabled=True, prefix="TEST", max_events=120, max_values=3)
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    handles = attach_module_trace_hooks(model, "seq", recurse=False, max_tensors=4)
    try:
        _ = model(torch.randn(5, 2))
    finally:
        remove_trace_hooks(handles)

    lines = _capture_lines(capsys)
    assert any("seq.inputs[0]" in line for line in lines)
    assert any("seq.outputs" in line for line in lines)


def test_trace_tensor_delta_prints_delta_summary(capsys) -> None:
    """Trace a concise tensor delta summary."""
    configure_trace(enabled=True, prefix="TEST", max_events=10, max_values=3)
    before = torch.tensor([1.0, -2.0, 3.0])
    after = torch.tensor([2.0, -1.0, 5.0])

    trace_tensor_delta("delta", before, after)
    lines = _capture_lines(capsys)

    assert any("delta:" in line for line in lines)
    assert any("max_abs=" in line for line in lines)
    assert any("mean_abs=" in line for line in lines)


def test_spflow_utils_debug_exports_trace_helpers() -> None:
    """Ensure trace helpers are exported from ``spflow.utils.debug``."""
    assert callable(trace_tensor_tree)
