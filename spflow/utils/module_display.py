"""Module visualization and display utilities for SPFlow modules.

This module provides utilities for converting SPFlow modules to readable string
representations with support for tree and PyTorch formats.

The Module class includes a to_str() method that uses this utility internally.

Example:
    >>> # Using the module method (recommended)
    >>> print(my_model.to_str())  # Tree view (default)
    >>> print(my_model.to_str(format="pytorch"))  # PyTorch repr format

    >>> # Or using the function directly
    >>> from spflow.utils.module_display import module_to_str
    >>> print(module_to_str(my_model))
"""

from __future__ import annotations

from typing import Optional, Any

from spflow.modules.module import Module

# Tree view constants
PREFIX_CHILD = "├─ "
PREFIX_LAST = "└─ "
PREFIX_CONTINUATION = "│  "
PREFIX_EMPTY = "   "


def module_to_str(
    module: Module,
    format: str = "tree",
    max_depth: Optional[int] = None,
    show_params: bool = True,
    show_scope: bool = True,
    _depth: int = 0,
    _is_last: bool = True,
    _prefix: str = "",
) -> str:
    """
    Convert an SPFlow module to a readable string representation.

    This function provides visualization formats for SPFlow modules,
    making it easier to inspect model architecture and understand hierarchical
    structure.

    Args:
        module: The SPFlow module to display.
        format: Visualization format, one of:
            - "tree": ASCII tree view (default, recommended)
            - "pytorch": Default PyTorch format
        max_depth: Maximum depth to display (None = unlimited). Only applies to tree format.
        show_params: Whether to show parameter shapes for Sum/Product nodes. Only applies to tree format.
        show_scope: Whether to show scope information. Only applies to tree format.
        _depth: Internal recursion depth (don't set manually).
        _is_last: Internal flag for tree formatting (don't set manually).
        _prefix: Internal prefix for tree formatting (don't set manually).

    Returns:
        String representation of the module.

    Examples:
        >>> leaves = Normal(scope=Scope([0, 1]), out_channels=2)
        >>> sum_node = Sum(inputs=leaves, out_channels=3)
        >>> print(module_to_str(sum_node))  # Tree format (default)
        >>> print(module_to_str(sum_node, format="pytorch"))  # PyTorch format
        >>> print(module_to_str(sum_node, max_depth=2))  # Limit depth
    """
    if format == "pytorch":
        return repr(module)
    elif format == "tree":
        return _tree_view(module, max_depth, show_params, show_scope, _depth, _is_last, _prefix)
    else:
        raise ValueError(f"Unknown format: {format!r}. Must be one of 'tree' or 'pytorch'.")


def _tree_view(
    module: Module,
    max_depth: Optional[int],
    show_params: bool,
    show_scope: bool,
    depth: int,
    is_last: bool,
    prefix: str,
) -> str:
    """Generate tree view representation."""
    # Check depth limit
    if max_depth is not None and depth > max_depth:
        return ""

    lines = []

    # Build current node line
    if depth == 0:
        # Root node: no prefix
        current_prefix = ""
    else:
        current_prefix = prefix + (PREFIX_LAST if is_last else PREFIX_CHILD)

    module_line = _format_node(module, show_params, show_scope)
    lines.append(current_prefix + module_line)

    # Build prefix for children
    if depth == 0:
        child_prefix = ""
    else:
        child_prefix = prefix + (PREFIX_EMPTY if is_last else PREFIX_CONTINUATION)

    # Get children to display
    children = _get_module_children(module)

    if children and (max_depth is None or depth < max_depth):
        for i, (child_name, child_module) in enumerate(children):
            is_last_child = i == len(children) - 1
            child_output = _tree_view(
                child_module,
                max_depth,
                show_params,
                show_scope,
                depth + 1,
                is_last_child,
                child_prefix,
            )
            if child_output:
                lines.append(child_output)

    return "\n".join(lines)


def _format_node(module: Module, show_params: bool = True, show_scope: bool = True) -> str:
    """Format a single module node for tree view."""
    module_name = module.__class__.__name__
    properties = []

    # Add key properties
    properties.append(f"D={module.out_features}")
    properties.append(f"C={module.out_channels}")

    if hasattr(module, "num_repetitions") and module.num_repetitions is not None:
        properties.append(f"R={module.num_repetitions}")

    props_str = ", ".join(properties)

    # Add scope info if requested
    scope_info = ""
    if show_scope and hasattr(module, "scope"):
        scope_str = _format_scope(module.scope)
        if scope_str:
            scope_info = f" → scope: {scope_str}"

    # Add parameter info if requested
    param_info = ""
    if show_params:
        if hasattr(module, "weights_shape"):
            param_info = f" [weights: {module.weights_shape}]"

    return f"{module_name} [{props_str}]{param_info}{scope_info}"


def _get_module_children(module: Module) -> list[tuple[str, Module]]:
    """Get child modules, skipping Cat/ModuleList wrappers and non-Module objects."""
    children = []

    # Check for direct inputs attribute
    if hasattr(module, "inputs"):
        inputs = module.inputs
        if isinstance(inputs, Module):
            # Skip Cat wrapper, get its children
            if inputs.__class__.__name__ == "Cat" and hasattr(inputs, "inputs"):
                cat_inputs = inputs.inputs
                if hasattr(cat_inputs, "__iter__"):
                    for i, child in enumerate(cat_inputs):
                        if isinstance(child, Module):
                            children.append((f"inputs[{i}]", child))
                else:
                    if isinstance(inputs, Module):
                        children.append(("inputs", inputs))
            else:
                children.append(("inputs", inputs))

    # Check for root_node attribute (RAT-SPN modules)
    if module.__class__.__name__ == "RatSPN":
        root_node = module.root_node
        if root_node is not None and isinstance(root_node, Module):
            children.append(("root_node", root_node))

    return children


def _format_scope(scope: Any) -> str:
    """Format scope information as readable string."""
    if scope is None:
        return ""

    # Handle Scope object
    if hasattr(scope, "query"):
        query = scope.query
        if not query:
            return ""

        # Convert to list and sort
        query_list = sorted(list(query))

        # Check if it's a contiguous range
        if query_list and query_list[-1] - query_list[0] + 1 == len(query_list):
            # Contiguous range: show as min-max
            return f"{query_list[0]}-{query_list[-1]}"
        else:
            # Non-contiguous: show as set
            return "{" + ", ".join(str(i) for i in query_list) + "}"

    return ""
