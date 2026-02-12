"""Tests for visualization utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.leaves import Normal
from spflow.modules.ops.cat import Cat
from spflow.modules.sums import Sum
from spflow.exceptions import GraphvizError, OptionalDependencyError
from spflow.utils import visualization
from spflow.utils.visualization import (
    _build_graph,
    _build_vis_label,
    _format_param_count,
    _format_scope_string,
    _get_module_color,
    visualize,
)


class DummyModule(Module):
    """Simple concrete module used to exercise visualization code paths."""

    def __init__(
        self,
        scope: Scope | None = None,
        inputs: Module | list[Module] | torch.nn.ModuleList | None = None,
        extra: str | None = None,
    ) -> None:
        super().__init__()
        self.scope = scope if scope is not None else Scope([0])
        self.in_shape = ModuleShape(1, 1, 1)
        self.out_shape = ModuleShape(1, 1, 1)
        if inputs is not None:
            self.inputs = inputs
        self._extra = extra

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([[self.scope]])

    def log_likelihood(self, data, cache=None):
        return torch.zeros((data.shape[0], 1, 1))

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.zeros((num_samples, 1))
        return data

    def _sample(self, data, sampling_ctx, cache, is_mpe: bool = False):
        del sampling_ctx
        del cache
        del is_mpe
        return data

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return self

    def _extra_vis_info(self) -> str | None:
        return self._extra


@pytest.fixture
def simple_model():
    """Create a simple model: Sum -> Normal leaf."""
    leaf = Normal(scope=Scope([0, 1]), out_channels=2)
    return Sum(inputs=leaf, out_channels=3)


@patch("spflow.utils.visualization.pydot")
def test_visualize_success(mock_pydot, simple_model, tmp_path):
    """Test successful visualization calls pydot methods."""
    # Setup mocks
    mock_graph = MagicMock()
    mock_pydot.Dot.return_value = mock_graph

    output_path = str(tmp_path / "test_graph")

    # Test different formats
    formats = ["png", "pdf", "svg", "dot", "plain", "canon"]
    for fmt in formats:
        mock_graph.reset_mock()
        visualize(simple_model, output_path, format=fmt)

        # Verify write methods called
        if fmt == "png":
            mock_graph.write_png.assert_called_with(f"{output_path}.png", prog="dot")
        elif fmt == "pdf":
            mock_graph.write_pdf.assert_called_with(f"{output_path}.pdf", prog="dot")
        elif fmt == "svg":
            mock_graph.write_svg.assert_called_with(f"{output_path}.svg", prog="dot")
        elif fmt == "dot":
            mock_graph.write_dot.assert_called_with(f"{output_path}.dot", prog="dot")
        elif fmt == "plain":
            mock_graph.write_plain.assert_called_with(f"{output_path}.plain", prog="dot")
        elif fmt == "canon":
            mock_graph.write.assert_called_with(f"{output_path}.{fmt}", format="canon", prog="dot")


@patch("spflow.utils.visualization.pydot")
def test_visualize_graphviz_not_found(mock_pydot, simple_model, tmp_path):
    """Test GraphvizError is raised when executable is not found."""
    mock_graph = MagicMock()
    mock_pydot.Dot.return_value = mock_graph

    # Mock write_png to raise FileNotFoundError (simulating missing binary)
    mock_graph.write_png.side_effect = FileNotFoundError("Executable not found")

    output_path = str(tmp_path / "test_graph")
    with pytest.raises(GraphvizError):
        visualize(simple_model, output_path, format="png")


@patch("spflow.utils.visualization.pydot")
def test_visualize_pydot_exception(mock_pydot, simple_model, tmp_path):
    """Test GraphvizError is raised on PydotException."""
    mock_graph = MagicMock()
    mock_pydot.Dot.return_value = mock_graph

    # Mock write_png to raise PydotException
    from pydot.exceptions import PydotException

    mock_graph.write_png.side_effect = PydotException("Error message")

    output_path = str(tmp_path / "test_graph")
    with pytest.raises(GraphvizError):
        visualize(simple_model, output_path, format="png")


def test_visualize_unsupported_format(simple_model, tmp_path):
    """Test ValueError for unsupported format."""
    output_path = str(tmp_path / "test_graph")
    with pytest.raises(ValueError):
        visualize(simple_model, output_path, format="invalid")


@patch("spflow.utils.visualization.pydot")
def test_visualize_engine_variants(mock_pydot, simple_model, tmp_path):
    """Test special engine handling (e.g., dot-lr)."""
    mock_graph = MagicMock()
    mock_pydot.Dot.return_value = mock_graph

    output_path = str(tmp_path / "test_graph")
    visualize(simple_model, output_path, format="png", engine="dot-lr")

    mock_pydot.Dot.assert_called_with(graph_type="digraph", rankdir="LR", dpi="300")
    # Should call write_png with prog="dot" (after internal conversion)
    mock_graph.write_png.assert_called_with(f"{output_path}.png", prog="dot")


def test_format_param_count_thresholds():
    """Test parameter formatting across size scales."""
    assert _format_param_count(42) == "42"
    assert _format_param_count(12_300) == "12.3K"
    assert _format_param_count(1_250_000) == "1.2M"


def test_format_scope_string_empty_and_ranges():
    """Test scope formatting for empty, short, and range runs."""
    assert _format_scope_string([]) == ""
    assert _format_scope_string([3, 1, 2, 2]) == "1...3"
    assert _format_scope_string([1, 3, 4]) == "1, 3, 4"


def test_build_vis_label_includes_extra_info():
    """Test that module-provided visualization info is appended."""
    module = DummyModule(extra="Extra: yes")
    label = _build_vis_label(module, show_shape=False, show_scope=False, show_params=False)
    assert label == "Extra: yes"


def test_get_module_color_leaf_is_green(monkeypatch):
    """Test leaf modules are colored as leaves."""
    from spflow.modules.leaves.leaf import LeafModule
    import spflow.modules.leaves as leaves_pkg

    monkeypatch.setattr(leaves_pkg, "LeafModule", LeafModule, raising=False)
    leaf = Normal(scope=Scope([0]), out_channels=1)
    assert _get_module_color(leaf) == visualization.Color.GREEN


def test_build_graph_returns_existing_node_when_visited():
    """Test visited nodes are returned without re-traversal."""
    graph = visualization.pydot.Dot(graph_type="digraph")
    module = DummyModule()
    node_id = _build_graph(module, graph, visited={id(module)})
    assert node_id == id(module)


def test_build_graph_skip_ops_modulelist_branch():
    """Test skip-ops traversal over ModuleList inputs."""
    leaf_a = Normal(scope=Scope([0]), out_channels=1)
    leaf_b = Normal(scope=Scope([1]), out_channels=1)
    cat = Cat(inputs=[leaf_a, leaf_b], dim=1)
    graph = visualization.pydot.Dot(graph_type="digraph")

    returned = _build_graph(cat, graph, parent_id=999, skip_ops=True)

    assert returned is None
    assert len(graph.get_edges()) == 2


def test_build_graph_skip_ops_list_and_single_module_branches():
    """Test skip-ops traversal over list and single-module inputs."""
    leaf_a = Normal(scope=Scope([0]), out_channels=1)
    leaf_b = Normal(scope=Scope([1]), out_channels=1)
    cat = Cat(inputs=[leaf_a, leaf_b], dim=1)
    graph_list = visualization.pydot.Dot(graph_type="digraph")
    cat._modules["inputs"] = [leaf_a, leaf_b]
    _build_graph(cat, graph_list, parent_id=777, skip_ops=True)
    assert len(graph_list.get_edges()) == 2

    graph_single = visualization.pydot.Dot(graph_type="digraph")
    cat._modules["inputs"] = leaf_a
    _build_graph(cat, graph_single, parent_id=778, skip_ops=True)
    assert len(graph_single.get_edges()) == 1


def test_build_graph_modulelist_and_list_inputs_for_regular_module():
    """Test traversal of ModuleList and list inputs on non-skipped modules."""
    leaf_a = DummyModule(scope=Scope([0]))
    leaf_b = DummyModule(scope=Scope([1]))

    parent_modulelist = DummyModule(inputs=torch.nn.ModuleList([leaf_a, leaf_b]))
    graph_modulelist = visualization.pydot.Dot(graph_type="digraph")
    _build_graph(parent_modulelist, graph_modulelist)
    assert len(graph_modulelist.get_edges()) == 2

    parent_list = DummyModule(inputs=[leaf_a, leaf_b])
    graph_list = visualization.pydot.Dot(graph_type="digraph")
    _build_graph(parent_list, graph_list)
    assert len(graph_list.get_edges()) == 2


def test_build_graph_root_node_branch_adds_edge():
    """Test RatSPN-style root_node traversal path."""
    root_owner = DummyModule()
    root_owner.root_node = DummyModule(scope=Scope([2]))
    graph = visualization.pydot.Dot(graph_type="digraph")
    _build_graph(root_owner, graph)
    assert len(graph.get_edges()) == 1


def test_visualization_import_error_path_executes_optional_dependency_message():
    """Execute module source with pydot import failing to cover dependency error path."""
    source_path = Path(__file__).resolve().parents[2] / "spflow" / "utils" / "visualization.py"
    source = source_path.read_text(encoding="utf-8")
    builtins_dict = dict(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__)
    original_import = builtins_dict["__import__"]

    def failing_import(name, *args, **kwargs):
        if name == "pydot" or name.startswith("pydot."):
            raise ImportError("forced missing pydot")
        return original_import(name, *args, **kwargs)

    builtins_dict["__import__"] = failing_import
    module_globals = {
        "__name__": "spflow.utils.visualization_no_pydot",
        "__file__": str(source_path),
        "__builtins__": builtins_dict,
    }

    with pytest.raises(OptionalDependencyError):
        exec(compile(source, str(source_path), "exec"), module_globals, module_globals)
