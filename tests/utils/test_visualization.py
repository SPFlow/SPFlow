"""Tests for visualization utilities."""

import pytest
from unittest.mock import MagicMock, patch
import torch

from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.sums import Sum
from spflow.exceptions import GraphvizError
from spflow.utils.visualization import visualize


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
    with pytest.raises(GraphvizError, match="Graphviz executable 'dot' not found"):
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
    with pytest.raises(GraphvizError, match="Error executing Graphviz"):
        visualize(simple_model, output_path, format="png")


def test_visualize_unsupported_format(simple_model, tmp_path):
    """Test ValueError for unsupported format."""
    output_path = str(tmp_path / "test_graph")
    with pytest.raises(ValueError, match="Unsupported format"):
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
