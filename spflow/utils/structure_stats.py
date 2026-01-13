"""Utilities for computing structure statistics of SPFlow circuits.

This module provides a lightweight, deterministic way to inspect model complexity
for debugging and experiment logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict

import torch

from spflow.exceptions import StructureError
from spflow.modules.module import Module


@dataclass(frozen=True, slots=True)
class StructureStats:
    """Structure statistics for a circuit rooted at a module.

    Attributes:
        num_nodes_total: Unique node count (includes root node).
        num_edges_total: Total structural edges (counts repeated incoming references).
        num_parameters_total: Total number of scalar parameters (unique tensors).
        node_type_counts: Counts by module class name.
        parameter_type_counts: Counts by parameter attribute name.
        max_depth: Longest root→leaf path length (leaf depth = 1).
        scope_size_min: Minimum scope size across visited nodes (or None).
        scope_size_max: Maximum scope size across visited nodes (or None).
        scope_size_mean: Mean scope size across visited nodes (or None).
        scope_size_histogram: Histogram of scope sizes across nodes.
        is_dag: True if shared nodes exist (graph reuse).
        num_shared_nodes: Number of nodes with in-degree > 1.
    """

    num_nodes_total: int
    num_edges_total: int
    num_parameters_total: int
    node_type_counts: dict[str, int]
    parameter_type_counts: dict[str, int]
    max_depth: int
    scope_size_min: int | None
    scope_size_max: int | None
    scope_size_mean: float | None
    scope_size_histogram: dict[int, int]
    is_dag: bool
    num_shared_nodes: int

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dict for logging."""
        return {
            "num_nodes_total": self.num_nodes_total,
            "num_edges_total": self.num_edges_total,
            "num_parameters_total": self.num_parameters_total,
            "node_type_counts": dict(self.node_type_counts),
            "parameter_type_counts": dict(self.parameter_type_counts),
            "max_depth": self.max_depth,
            "scope_size_min": self.scope_size_min,
            "scope_size_max": self.scope_size_max,
            "scope_size_mean": self.scope_size_mean,
            "scope_size_histogram": dict(self.scope_size_histogram),
            "is_dag": self.is_dag,
            "num_shared_nodes": self.num_shared_nodes,
        }


def _tensor_identity(tensor: torch.Tensor) -> tuple[object, ...]:
    """Return a best-effort identity for a tensor's underlying storage.

    Args:
        tensor: Tensor to identify.

    Returns:
        Tuple key suitable for de-duplication across shared parameters.
    """
    try:
        device_index = tensor.device.index if tensor.device.index is not None else -1
        return (tensor.device.type, device_index, int(tensor.data_ptr()))
    except Exception:
        return ("id", id(tensor))


def _iter_structure_children(module: Module) -> list[Module]:
    """Return structural children for statistics traversal.

    Mirrors the traversal behavior of ``model.to_str()``:
    - Skips ``Cat`` wrappers by yielding their inputs directly.
    - Expands ``nn.ModuleList`` inputs into individual child modules.
    - Includes wrapped modules via the ``Wrapper.module`` attribute.
    - Includes ``RatSPN.root_node`` as a child.

    Args:
        module: Module whose children should be returned.

    Returns:
        List of child modules.
    """
    children: list[Module] = []

    wrapped = getattr(module, "module", None)
    if isinstance(wrapped, Module):
        children.append(wrapped)

    try:
        inputs = module.inputs
    except Exception:
        inputs = None

    if inputs is not None:
        # Handle nn.ModuleList without importing torch.nn here (type is stable by name).
        if hasattr(inputs, "__iter__") and inputs.__class__.__name__ == "ModuleList":
            for child in inputs:
                if isinstance(child, Module):
                    children.append(child)
        elif isinstance(inputs, list):
            for child in inputs:
                if isinstance(child, Module):
                    children.append(child)
        elif isinstance(inputs, Module):
            if inputs.__class__.__name__ == "Cat" and hasattr(inputs, "inputs"):
                cat_inputs = getattr(inputs, "inputs", None)
                if cat_inputs is not None and hasattr(cat_inputs, "__iter__"):
                    for child in cat_inputs:
                        if isinstance(child, Module):
                            children.append(child)
                elif isinstance(cat_inputs, Module):
                    children.append(cat_inputs)
            else:
                children.append(inputs)

    if module.__class__.__name__ == "RatSPN":
        root_node = getattr(module, "root_node", None)
        if isinstance(root_node, Module):
            children.append(root_node)

    return children


def get_structure_stats(model: Module) -> StructureStats:
    """Compute structure statistics for a circuit rooted at ``model``.

    The traversal is DAG-aware:
    - Nodes are counted uniquely (by object identity).
    - Edges count every parent→child reference (including multiple incoming
      references for shared subgraphs).
    - Parameters are counted uniquely across shared subgraphs.

    Args:
        model: Root module of the circuit.

    Returns:
        A :class:`StructureStats` instance.

    Raises:
        StructureError: If a structural cycle is detected while computing depth.
    """
    visited: set[int] = set()
    stack: list[Module] = [model]

    node_type_counts: Counter[str] = Counter()
    parameter_type_counts: Counter[str] = Counter()

    in_degree: defaultdict[int, int] = defaultdict(int)
    num_edges_total = 0

    unique_parameter_numel: dict[tuple[object, ...], int] = {}
    scope_sizes: list[int] = []

    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        node_type_counts[node.__class__.__name__] += 1

        if hasattr(node, "scope"):
            try:
                scope_sizes.append(len(node.scope))
            except Exception:
                pass

        for param_name, param in node.named_parameters(recurse=False):
            if param is None:
                continue
            parameter_type_counts[param_name] += 1
            key = _tensor_identity(param)
            if key not in unique_parameter_numel:
                unique_parameter_numel[key] = int(param.numel())

        children = _iter_structure_children(node)
        num_edges_total += len(children)
        for child in children:
            in_degree[id(child)] += 1
            if id(child) not in visited:
                stack.append(child)

    num_shared_nodes = sum(1 for degree in in_degree.values() if degree > 1)

    def _max_depth(root: Module) -> int:
        memo: dict[int, int] = {}
        visiting: set[int] = set()

        def dfs(node: Module) -> int:
            node_id = id(node)
            if node_id in memo:
                return memo[node_id]
            if node_id in visiting:
                raise StructureError("Cycle detected while computing structure depth.")
            visiting.add(node_id)
            children = _iter_structure_children(node)
            if not children:
                depth = 1
            else:
                depth = 1 + max(dfs(child) for child in children)
            visiting.remove(node_id)
            memo[node_id] = depth
            return depth

        return dfs(root)

    scope_size_min = min(scope_sizes) if scope_sizes else None
    scope_size_max = max(scope_sizes) if scope_sizes else None
    scope_size_mean = (sum(scope_sizes) / len(scope_sizes)) if scope_sizes else None
    scope_size_histogram = dict(sorted(Counter(scope_sizes).items()))

    return StructureStats(
        num_nodes_total=len(visited),
        num_edges_total=num_edges_total,
        num_parameters_total=sum(unique_parameter_numel.values()),
        node_type_counts=dict(sorted(node_type_counts.items())),
        parameter_type_counts=dict(sorted(parameter_type_counts.items())),
        max_depth=_max_depth(model),
        scope_size_min=scope_size_min,
        scope_size_max=scope_size_max,
        scope_size_mean=scope_size_mean,
        scope_size_histogram=scope_size_histogram,
        is_dag=num_shared_nodes > 0,
        num_shared_nodes=num_shared_nodes,
    )


def structure_stats_to_str(stats: StructureStats, max_node_types: int = 10) -> str:
    """Format structure statistics as a human-readable, deterministic text summary.

    Args:
        stats: StructureStats instance.
        max_node_types: Maximum number of node types to include.

    Returns:
        Multi-line string summary.
    """
    lines: list[str] = []
    lines.append("Structure statistics")
    lines.append(f"- nodes: {stats.num_nodes_total}")
    lines.append(f"- edges: {stats.num_edges_total}")
    lines.append(f"- parameters: {stats.num_parameters_total}")
    lines.append(f"- max_depth: {stats.max_depth}")
    lines.append(f"- shared_nodes: {stats.num_shared_nodes}")

    if stats.scope_size_min is not None:
        mean_str = f"{stats.scope_size_mean:.3f}" if stats.scope_size_mean is not None else "None"
        lines.append(f"- scope_size: min={stats.scope_size_min} max={stats.scope_size_max} mean={mean_str}")

    if stats.node_type_counts:
        sorted_types = sorted(
            stats.node_type_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        shown = ", ".join(f"{name}={count}" for name, count in sorted_types[:max_node_types])
        lines.append(f"- node_types: {shown}")
        if len(sorted_types) > max_node_types:
            lines.append(f"- node_types_more: {len(sorted_types) - max_node_types}")

    return "\n".join(lines)
