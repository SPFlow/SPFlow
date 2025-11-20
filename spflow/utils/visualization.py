"""Graph visualization utilities for SPFlow modules.

This module provides functions to visualize SPFlow module graphs using pydot and graphviz.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

from spflow.exceptions import GraphvizError

try:
    import pydot
    from pydot.exceptions import PydotException
except ImportError as e:
    raise ImportError(
        "The 'pydot' package is required for visualization functionality.\n\n"
        "To install pydot and graphviz dependencies:\n"
        "  1. Install the Graphviz system dependency:\n"
        "     - On macOS: brew install graphviz\n"
        "     - On Ubuntu/Debian: sudo apt-get install graphviz\n"
        "     - On Windows: Download from https://graphviz.org/download/\n"
        "  2. Install pydot Python package:\n"
        "     - pip install pydot\n"
        "     - OR install SPFlow with visualization extras: pip install spflow[viz]\n\n"
        "For more details, see the README.md file in the SPFlow repository."
    ) from e

if TYPE_CHECKING:
    from spflow.modules.base import Module

from spflow.modules.leaves.base import LeafModule
from spflow.modules.ops.cat import Cat
from spflow.modules.ops.split import Split
from spflow.modules.ops.split_alternate import SplitAlternate
from spflow.modules.ops.split_halves import SplitHalves


class Color(str, Enum):
    """Tab10 colormap colors for module visualization.

    Uses matplotlib's tab10 colormap for consistent, perceptually uniform colors.
    Derived from: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """

    # Tab10 colors (indexed 0-9)
    BLUE = "#1f77b4"  # tab10[0] - Sum-related modules
    ORANGE = "#ff7f0e"  # tab10[1] - Product-related modules
    GREEN = "#2ca02c"  # tab10[2] - Leaf modules
    RED = "#d62728"  # tab10[3] - RatSPN
    PURPLE = "#9467bd"  # tab10[4] - Split-related modules
    BROWN = "#8c564b"  # tab10[5] - Factorize
    PINK = "#e377c2"  # tab10[6] - Cat
    GRAY = "#7f7f7f"  # tab10[7] - Default/Unknown types


# Ops modules to skip in visualization (pass-through/helper modules)
# When these modules are encountered, they are bypassed and their inputs are connected
# directly to the parent module
SKIP_OPS = {Cat, Split, SplitHalves, SplitAlternate}


def _format_param_count(count: int) -> str:
    """Format parameter count with K/M suffixes for readability.

    Args:
        count: Number of parameters.

    Returns:
        Formatted string (e.g., "1.2K", "3.5M", "42").
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"


def _count_parameters(module: Module) -> int:
    """Count parameters for a module.

    For leaves modules, counts all parameters (including the distribution child module).
    For other modules, counts only parameters directly owned by this module (excluding children).

    Args:
        module: The module to count parameters for.

    Returns:
        Number of parameters. For leaves modules, includes child distribution parameters.
        For other modules, only direct parameters.
    """
    # For leaves modules, include all parameters (including distribution child)
    if isinstance(module, LeafModule):
        return sum(p.numel() for p in module.parameters())
    # For other modules, only count direct parameters
    return sum(p.numel() for p in module.parameters(recurse=False))


def visualize(
    module: Module,
    output_path: str,
    show_scope: bool = True,
    show_shape: bool = True,
    show_params: bool = True,
    format: str = "pdf",
    dpi: int = 300,
    engine: str = "dot",
    rankdir: Literal["TB", "LR", "BT", "RL"] = "BT",
    node_shape: str = "box",
    skip_ops: bool = True,
) -> None:
    """Visualize a SPFlow module as a directed graph and save to file.

    Args:
        module: The root module to visualize.
        output_path: Path to save the visualization (without extension).
        show_scope: Whether to display scope information in node labels.
        show_shape: Whether to display shape information (D: out_features, C: out_channels) in node labels.
        show_params: Whether to display parameter count in node labels. Parameter counts are formatted
            with K/M suffixes for readability (e.g., "1.2K", "3.5M").
        format: Output format - 'png', 'pdf', 'svg', 'dot', 'plain', or 'canon'.
            Text-based formats are useful for viewing graph structure in the terminal:
            - 'dot'/'canon': Graphviz DOT language source code
            - 'plain': Simple text format with node positions and edges
        dpi: DPI for rasterized formats (png). Applied via graph-level dpi attribute.
        engine: Graphviz layout engine. Options:
            - 'dot' (default): Hierarchical top-down layout, best for directed acyclic graphs
            - 'dot-lr': Hierarchical left-right layout (automatically sets rankdir='LR')
            - 'neato': Spring model layout (force-directed)
            - 'fdp': Force-directed placement, similar to neato
            - 'circo': Circular layout
            - 'twopi': Radial layout
            - 'osage': Clustered layout
        rankdir: Direction of graph layout (only used with 'dot' and 'dot-lr' engines):
            - 'TB': Top to bottom (default)
            - 'LR': Left to right
            - 'BT': Bottom to top
            - 'RL': Right to left
        node_shape: Shape of nodes. Common options: 'box' (default), 'circle', 'ellipse',
            'diamond', 'triangle', 'plaintext', 'record', 'Mrecord'.
        skip_ops: Whether to skip ops modules in visualization (Cat, Split, SplitHalves, SplitAlternate).
            These are pass-through modules that are bypassed and their inputs connected directly to parent.
            Defaults to True.

    Returns:
        None. The visualization is saved to the specified output path.

    """
    # Handle special engine variants
    if engine == "dot-lr":
        engine = "dot"
        rankdir = "LR"

    # Create the pydot graph
    graph = pydot.Dot(graph_type="digraph", rankdir=rankdir, dpi=str(dpi))

    # Set graph attributes for better aesthetics
    graph.set_graph_defaults(
        fontname="Helvetica",
        fontsize="11",
        nodesep="0.5",
        ranksep="0.8",
    )

    # Set node defaults
    graph.set_node_defaults(
        shape=node_shape,
        style="rounded,filled",
        fillcolor="white",
        fontname="Helvetica",
        fontsize="11",
        penwidth="2.5",
        margin="0.15,0.08",  # Horizontal, vertical padding
    )

    # Set edge defaults
    graph.set_edge_defaults(
        color="#333333",
        penwidth="2.0",
        arrowsize="0.8",
    )

    # Build the graph
    _build_graph(
        module,
        graph,
        show_scope=show_scope,
        show_shape=show_shape,
        show_params=show_params,
        skip_ops=skip_ops,
    )

    # Generate output file
    output_file = f"{output_path}.{format}"

    # Write output using the specified engine
    try:
        match format:
            case "png":
                graph.write_png(output_file, prog=engine)
            case "pdf":
                graph.write_pdf(output_file, prog=engine)
            case "svg":
                graph.write_svg(output_file, prog=engine)
            case "dot":
                graph.write_dot(output_file, prog=engine)
            case "plain":
                graph.write_plain(output_file, prog=engine)
            case "canon":
                graph.write(output_file, format="canon", prog=engine)
            case _:
                raise ValueError(
                    f"Unsupported format: {format}. Supported formats: png, pdf, svg, dot, plain, canon"
                )

    except FileNotFoundError as e:
        # This error occurs when Graphviz is not installed or not in PATH
        raise GraphvizError(
            f"Graphviz executable '{engine}' not found. This usually means Graphviz is not installed or not in your system PATH."
        ) from e
    except (AssertionError, OSError, PydotException) as e:
        # Catch errors from pydot/graphviz execution
        raise GraphvizError(
            f"Error executing Graphviz: {str(e)}\n\n"
            f"This error typically indicates a problem with your Graphviz installation."
        ) from e


def _build_graph(
    module: Module,
    graph: pydot.Dot,
    show_scope: bool = False,
    show_shape: bool = False,
    show_params: bool = False,
    visited: set | None = None,
    parent_id: int | None = None,
    skip_ops: bool = True,
) -> int | None:
    """Recursively build a pydot graph from a module tree.

    Args:
        module: Current module to add to the graph.
        graph: pydot Dot graph to populate.
        show_scope: Whether to include scope information in labels.
        show_shape: Whether to include shape information in labels.
        show_params: Whether to include parameter counts in labels.
        visited: Set of module IDs already visited (to avoid duplicates).
        parent_id: ID of the parent node (used when skipping modules).
        skip_ops: Whether to skip ops modules in visualization.

    Returns:
        The node ID for the current module, or None if the module was skipped.
    """
    from torch import nn

    if visited is None:
        visited = set()

    node_id = id(module)

    # Check if this module should be skipped in the visualization
    if skip_ops and SKIP_OPS and isinstance(module, tuple(SKIP_OPS)) and parent_id is not None:
        # This is a pass-through module - skip it and connect its inputs directly to parent
        if hasattr(module, "inputs"):
            inputs = module.inputs
            # Handle nn.ModuleList (from Cat module)
            if isinstance(inputs, nn.ModuleList):
                inputs = list(inputs)

            if isinstance(inputs, list):
                # Multiple inputs - recursively add each
                for input_module in inputs:
                    child_id = _build_graph(
                        input_module, graph, show_scope, show_shape, show_params, visited, parent_id, skip_ops
                    )
                    # Only add edge if child was actually added to graph (not skipped)
                    if child_id is not None:
                        edge = pydot.Edge(str(child_id), str(parent_id))
                        graph.add_edge(edge)
            else:
                # Single input - recursively add it
                child_id = _build_graph(
                    inputs, graph, show_scope, show_shape, show_params, visited, parent_id, skip_ops
                )
                # Only add edge if child was actually added to graph (not skipped)
                if child_id is not None:
                    edge = pydot.Edge(str(child_id), str(parent_id))
                    graph.add_edge(edge)
        return None  # Return None to indicate this module was skipped

    # Skip if already visited
    if node_id in visited:
        return node_id

    visited.add(node_id)

    # Create node label
    label = _get_module_label(module, show_scope=show_scope, show_shape=show_shape, show_params=show_params)

    # Get color for this module type
    color = _get_module_color(module)

    # Add node to graph
    node = pydot.Node(
        str(node_id),
        label=label,
        color=color,
    )
    graph.add_node(node)

    # Traverse inputs if they exist
    if hasattr(module, "inputs"):
        inputs = module.inputs
        # Handle nn.ModuleList (from Cat module)
        if isinstance(inputs, nn.ModuleList):
            inputs = list(inputs)

        if isinstance(inputs, list):
            # Multiple inputs
            for input_module in inputs:
                child_id = _build_graph(
                    input_module,
                    graph,
                    show_scope,
                    show_shape,
                    show_params,
                    visited,
                    parent_id=node_id,
                    skip_ops=skip_ops,
                )
                # Only add edge if child was actually added to graph (not skipped)
                if child_id is not None:
                    edge = pydot.Edge(str(child_id), str(node_id))
                    graph.add_edge(edge)
        else:
            # Single input
            child_id = _build_graph(
                inputs,
                graph,
                show_scope,
                show_shape,
                show_params,
                visited,
                parent_id=node_id,
                skip_ops=skip_ops,
            )
            # Only add edge if child was actually added to graph (not skipped)
            if child_id is not None:
                edge = pydot.Edge(str(child_id), str(node_id))
                graph.add_edge(edge)

    # Special handling for RatSPN: traverse through root_node
    if hasattr(module, "root_node"):
        child_id = _build_graph(
            module.root_node,
            graph,
            show_scope,
            show_shape,
            show_params,
            visited,
            parent_id=node_id,
            skip_ops=skip_ops,
        )
        # Only add edge if child was actually added to graph (not skipped)
        if child_id is not None:
            edge = pydot.Edge(str(child_id), str(node_id))
            graph.add_edge(edge)

    return node_id


def _format_scope_string(scopes: list[int]) -> str:
    """Format a list of scope indices, using ranges for consecutive sequences.

    Consecutive sequences of 3 or more indices are represented as ranges (e.g., "0...4").
    Shorter consecutive sequences are listed individually.

    Args:
        scopes: List of scope indices.

    Returns:
        Formatted scope string with ranges for consecutive sequences.
    """
    if not scopes:
        return ""

    # Sort and deduplicate
    sorted_scopes = sorted(set(scopes))

    result = []
    i = 0

    while i < len(sorted_scopes):
        start = sorted_scopes[i]
        end = start

        # Find the end of the consecutive sequence
        while i + 1 < len(sorted_scopes) and sorted_scopes[i + 1] == sorted_scopes[i] + 1:
            i += 1
            end = sorted_scopes[i]

        # Determine if we should use a range (4 or more consecutive)
        sequence_length = end - start + 1
        if sequence_length >= 3:
            # Use range for 3+ consecutive numbers
            result.append(f"{start}...{end}")
        else:
            # List individually for 1-3 numbers
            result.append(", ".join(str(x) for x in range(start, end + 1)))

        i += 1

    return ", ".join(result)


def _get_module_label(
    module: Module, show_scope: bool = False, show_shape: bool = False, show_params: bool = False
) -> str:
    """Generate a label for a module node.

    Uses HTML labels with bold module names for better visibility.

    Args:
        module: The module to generate a label for.
        show_scope: Whether to include scope information.
        show_shape: Whether to include shape information (for Sum nodes: C-in and C-out, for others: F and C).
        show_params: Whether to include parameter count.

    Returns:
        An HTML label string for the module (compatible with pydot/graphviz).
    """
    # Get the class name
    class_name = module.__class__.__name__

    # Start with bold module name using HTML label
    label_parts = [f"<B>{class_name}</B>"]

    # Add shape information if requested
    if show_shape:
        # Specialized labels for different module types
        if class_name == "Sum":
            c_in = module._in_channels_total
            c_out = module._out_channels_total
            d = module.out_features
            label = f"D: {d}, C-in: {c_in}, C-out: {c_out}"
            if hasattr(module, "num_repetitions"):
                label += f", R: {module.num_repetitions}"
            label_parts.append(label)
        elif class_name == "ElementwiseSum":
            c_per_in = module._in_channels_per_input
            c_out = module._num_sums
            d = module.out_features
            label = f"D: {d}, C-per-in: {c_per_in}, C-out: {c_out}"
            if hasattr(module, "num_repetitions"):
                label += f", R: {module.num_repetitions}"
            label_parts.append(label)
        elif class_name == "MixingLayer":
            c_in = module._in_channels
            c_out = module.out_channels
            label_parts.append(f"C-in: {c_in} (reps), C-out: {c_out}")
        elif class_name == "Product":
            d = module.out_features
            c = module.out_channels
            label_parts.append(f"D: {d}, C: {c}")
        elif class_name == "ElementwiseProduct":
            f = module.out_features
            c = module.out_channels
            label_parts.append(f"D: {f}, C: {c}")
        elif class_name == "OuterProduct":
            c_in = module._max_out_channels
            c_out = module.out_channels
            label_parts.append(f"C-in: {c_in}, C-out: {c_out}")
        else:
            # For other modules, show out_features and out_channels
            out_features = module.out_features
            out_channels = module.out_channels
            label_parts.append(f"D: {out_features}, C: {out_channels}")

    # Add parameter count if requested (only if > 0 to avoid clutter for modules like Product)
    if show_params:
        param_count = _count_parameters(module)
        if param_count > 0:
            formatted_count = _format_param_count(param_count)
            label_parts.append(f"Params: {formatted_count}")

    # Add scope information if requested
    if show_scope:
        scope_str = _format_scope_string(sorted(module.scope.query))
        label_parts.append(f"Scope: {scope_str}")

    # Use <br/> for line breaks in HTML labels, and wrap entire label in angle brackets
    return "<" + "<br/>".join(label_parts) + ">"


def _get_module_color(module: Module) -> str:
    """Get the color for a module based on its type.

    Uses matplotlib tab10 colormap for consistent, distinguishable colors.
    Related module types share colors within groups.

    Args:
        module: The module to get a color for.

    Returns:
        A color string (hex) for the module type based on tab10 colormap.
    """
    class_name = module.__class__.__name__

    # Check if this is a leaves module (all leaves modules get the same color)
    try:
        from spflow.modules.leaves import LeafModule

        if isinstance(module, LeafModule):
            return Color.GREEN
    except ImportError:
        # Fallback to class name checking if LeafModule can't be imported
        leaf_modules = {
            "Normal",
            "Categorical",
            "Bernoulli",
            "Poisson",
            "Exponential",
            "CondNormal",
            "CondCategorical",
        }
        if class_name in leaf_modules:
            return Color.GREEN

    # Define color mapping for different module types using Color enum
    color_map = {
        # Sum modules
        "Sum": Color.BLUE,
        "ElementwiseSum": Color.BLUE,
        "MixingLayer": Color.BLUE,
        # Product modules
        "Product": Color.ORANGE,
        "ElementwiseProduct": Color.ORANGE,
        "OuterProduct": Color.ORANGE,
        # Operations - Cat
        "Cat": Color.PINK,
        # Operations - Split
        "Split": Color.PURPLE,
        "SplitHalves": Color.PURPLE,
        "SplitAlternate": Color.PURPLE,
        # Operations - Factorize
        "Factorize": Color.BROWN,
        # RAT-SPN
        "RatSPN": Color.RED,
    }

    # Return color if found, otherwise use GRAY as default
    return color_map.get(class_name, Color.GRAY)
