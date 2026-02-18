"""Utility functions and helper classes for SPFlow.

This module provides various utility functions and helper classes that support
the main functionality of SPFlow. These include model management, visualization,
sampling contexts, caching mechanisms, and other supporting utilities.
These utilities are designed to work seamlessly with the main SPFlow modules
while maintaining clean separation of concerns.
"""

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
from spflow.utils.domain import DataType, Domain
from spflow.utils.histogram import get_bin_edges_torch
from spflow.utils.range_inference import log_likelihood_interval
from spflow.utils.replace import replace

__all__ = [
    "replace",
    "DataType",
    "Domain",
    "get_bin_edges_torch",
    "log_likelihood_interval",
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
