"""Backward-compatible APC imports for debug tracing utilities."""

from spflow.utils.debug import (
    attach_module_trace_hooks,
    configure_trace,
    remove_trace_hooks,
    trace_module_io,
    trace_module_state,
    trace_sampling_context,
    trace_tensor,
    trace_tensor_delta,
    trace_tensor_tree,
    trace_value,
)

__all__ = [
    "configure_trace",
    "trace_tensor",
    "trace_value",
    "trace_tensor_delta",
    "trace_tensor_tree",
    "trace_module_state",
    "trace_module_io",
    "attach_module_trace_hooks",
    "remove_trace_hooks",
    "trace_sampling_context",
]
