"""Tests for the graph visualization utilities."""

import errno
import os
import shutil
import tempfile
from unittest.mock import patch

import pydot
import pytest

from spflow.exceptions import GraphvizError
from spflow.meta import Scope
from spflow.modules.leaves import Categorical, Normal
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.visualization import (
    visualize,
    _build_graph,
)


def has_graphviz_dot():
    """Check if the graphviz 'dot' binary is available on the system."""
    return shutil.which("dot") is not None


# Skip all visualization tests that require graphviz if 'dot' binary is not available
pytestmark_requires_graphviz = pytest.mark.skipif(
    not has_graphviz_dot(),
    reason="graphviz 'dot' binary not found. Install graphviz to run visualization tests.",
)


class TestVisualizationSkipModules:
    """Test that pass-through modules (Cat, Split, etc.) are skipped in visualization."""

    pytestmark = pytestmark_requires_graphviz

    def test_product_with_multiple_inputs_skips_cat(self):
        """Test that Cat module is skipped when Product has multiple inputs.

        When Product([Sum, Sum]) is created, internally it creates a Cat module.
        The visualization should skip this Cat module and show Sum modules directly
        as inputs to Product.
        """
        # Create two Sum modules with disjoint scopes (required for Product)
        leaf1 = Normal(scope=Scope([0]), out_channels=2)
        leaf2 = Normal(scope=Scope([1]), out_channels=2)
        sum1 = Sum(inputs=leaf1, out_channels=2)
        sum2 = Sum(inputs=leaf2, out_channels=2)

        # Create Product with multiple inputs (internally creates Cat)
        product = Product(inputs=[sum1, sum2])

        # Build the graph
        graph = pydot.Dot(graph_type="digraph")
        _build_graph(product, graph)

        # Get all nodes
        nodes = graph.get_nodes()
        # Filter out the default node (pydot creates a default node entry)
        nodes = [n for n in nodes if n.get_name() not in ("node", "edge", "graph")]

        # Check that no node represents a Cat module by examining the module structure
        # We can't directly check the module attribute in pydot, so we verify the structure
        # The graph should have exactly 5 nodes: Product, 2 Sums, 2 Normals (no Cat)
        assert len(nodes) == 5, f"Expected 5 nodes (Product, 2 Sums, 2 Normals), got {len(nodes)}"

        # Check that Product has 2 direct inputs (Sum nodes)
        product_id = str(id(product))
        edges = graph.get_edges()
        edges_to_product = [e for e in edges if e.get_destination() == product_id]
        assert len(edges_to_product) == 2, "Product should have 2 direct inputs (Sum nodes)"

    def test_single_input_product_does_not_create_cat(self):
        """Test that Product with single input does not create Cat module."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        sum_node = Sum(inputs=leaf, out_channels=2)

        # Create Product with single input (should not create Cat)
        product = Product(inputs=sum_node)

        # Build the graph
        graph = pydot.Dot(graph_type="digraph")
        _build_graph(product, graph)

        # Get all nodes
        nodes = graph.get_nodes()
        nodes = [n for n in nodes if n.get_name() not in ("node", "edge", "graph")]

        # The graph should have exactly 3 nodes: Product, Sum, Normal (no Cat)
        assert len(nodes) == 3, f"Expected 3 nodes (Product, Sum, Normal), got {len(nodes)}"

        # Check that Sum is directly connected to Product
        product_id = str(id(product))
        edges = graph.get_edges()
        edges_to_product = [e for e in edges if e.get_destination() == product_id]
        assert len(edges_to_product) == 1, "Product should have 1 direct input (Sum node)"


class TestVisualizationBasics:
    """Test basic visualization functionality."""

    pytestmark = pytestmark_requires_graphviz

    @pytest.fixture
    def simple_model(self):
        """Create a simple model: Sum -> Normal leaves."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        return Sum(inputs=leaf, out_channels=3)

    def test_visualize_creates_output_file(self, simple_model):
        """Test that visualize_module creates output file with correct extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_model")
            visualize(simple_model, output_path, format="png")
            assert os.path.exists(f"{output_path}.png")

    def test_invalid_format_raises_error(self, simple_model):
        """Test that invalid format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_model")
            with pytest.raises(ValueError, match="Unsupported format"):
                visualize(simple_model, output_path, format="invalid_format")


class TestVisualizationOptions:
    """Test visualization customization options."""

    pytestmark = pytestmark_requires_graphviz

    @pytest.fixture
    def model_with_scope(self):
        """Create a model with specific scopes."""
        leaf1 = Normal(scope=Scope([0, 1]), out_channels=2)
        leaf2 = Normal(scope=Scope([2, 3]), out_channels=2)
        return Product(inputs=[leaf1, leaf2])

    def test_visualize_with_options(self, model_with_scope):
        """Test that visualization accepts and applies customization options.

        Verify that show_scope, show_shape, show_params, dpi, engine, rankdir parameters
        are accepted without error and produce valid output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_with_options")
            # Test that all options are accepted and don't raise errors
            visualize(
                model_with_scope,
                output_path,
                show_scope=True,
                show_shape=True,
                show_params=True,
                dpi=150,
                engine="dot",
                rankdir="LR",
                format="png",
            )
            assert os.path.exists(f"{output_path}.png")


class TestVisualizationComplexStructures:
    """Test visualization with complex module structures."""

    pytestmark = pytestmark_requires_graphviz

    def test_product_of_sums_graph_structure(self):
        """Test that Product of Sum nodes creates correct graph structure."""
        leaf1 = Normal(scope=Scope([0]), out_channels=2)
        leaf2 = Normal(scope=Scope([1]), out_channels=2)
        sum1 = Sum(inputs=leaf1, out_channels=2)
        sum2 = Sum(inputs=leaf2, out_channels=2)
        product = Product(inputs=[sum1, sum2])

        # Build graph directly to test structure (not rendering)
        graph = pydot.Dot(graph_type="digraph")
        _build_graph(product, graph)
        nodes = [n for n in graph.get_nodes() if n.get_name() not in ("node", "edge", "graph")]

        # Should have Product, 2 Sums, 2 Normals
        assert len(nodes) == 5

    def test_deeply_nested_structure_graph_structure(self):
        """Test deeply nested module structure creates correct graph."""
        leaf = Normal(scope=Scope([0]), out_channels=2)
        level1 = Sum(inputs=leaf, out_channels=2)
        leaf2 = Normal(scope=Scope([1]), out_channels=2)
        product = Product(inputs=[level1, leaf2])
        level3 = Sum(inputs=product, out_channels=2)

        # Build graph directly to test structure
        graph = pydot.Dot(graph_type="digraph")
        _build_graph(level3, graph)
        nodes = [n for n in graph.get_nodes() if n.get_name() not in ("node", "edge", "graph")]

        # Should have Sum, Product, Sum, Normal, Normal
        assert len(nodes) == 5


class TestVisualizationEdgeCases:
    """Test edge cases and special scenarios."""

    pytestmark = pytestmark_requires_graphviz

    def test_single_leaf_module_graph(self):
        """Test visualization with a single leaves module creates valid graph."""
        leaf = Normal(scope=Scope([0, 1, 2]), out_channels=3)

        # Build graph to verify structure
        graph = pydot.Dot(graph_type="digraph")
        _build_graph(leaf, graph)
        nodes = [n for n in graph.get_nodes() if n.get_name() not in ("node", "edge", "graph")]

        # Should have at least the Normal node
        assert len(nodes) > 0

    def test_large_scope_graph(self):
        """Test visualization with large scope creates valid graph."""
        leaf = Normal(scope=Scope(list(range(10))), out_channels=2)
        model = Sum(inputs=leaf, out_channels=3)

        # Build graph to verify structure
        graph = pydot.Dot(graph_type="digraph")
        _build_graph(model, graph)
        nodes = [n for n in graph.get_nodes() if n.get_name() not in ("node", "edge", "graph")]

        # Should have Sum and Normal nodes
        assert len(nodes) == 2


class TestVisualizationIntegration:
    """Integration tests with realistic models."""

    pytestmark = pytestmark_requires_graphviz

    def test_typical_spn_structure_graph(self):
        """Test visualization with a typical SPN structure creates correct graph."""
        # Create a small but realistic SPN
        leaf1 = Normal(scope=Scope([0]), out_channels=3)
        leaf2 = Normal(scope=Scope([1]), out_channels=3)
        leaf3 = Normal(scope=Scope([2]), out_channels=3)
        leaf4 = Normal(scope=Scope([3]), out_channels=3)

        prod1 = Product(inputs=[leaf1, leaf2])
        prod2 = Product(inputs=[leaf3, leaf4])

        sum1 = Sum(inputs=prod1, out_channels=2)
        sum2 = Sum(inputs=prod2, out_channels=2)

        root_prod = Product(inputs=[sum1, sum2])
        root = Sum(inputs=root_prod, out_channels=1)

        # Build graph to verify structure
        graph = pydot.Dot(graph_type="digraph")
        _build_graph(root, graph)
        nodes = [n for n in graph.get_nodes() if n.get_name() not in ("node", "edge", "graph")]

        # Should have 10 nodes: 1 root Sum, 1 Product, 2 Sums, 2 Products, 4 Normals
        assert len(nodes) == 10

    def test_mixed_complexity_graph(self):
        """Test visualization with mixed complexity model creates correct graph."""
        # Simple branch
        simple_leaf = Normal(scope=Scope([0]), out_channels=2)
        simple_sum = Sum(inputs=simple_leaf, out_channels=2)

        # Complex branch
        complex_leaf1 = Normal(scope=Scope([1]), out_channels=3)
        complex_leaf2 = Categorical(scope=Scope([2]), out_channels=3, K=5)
        complex_prod = Product(inputs=[complex_leaf1, complex_leaf2])
        complex_sum = Sum(inputs=complex_prod, out_channels=2)

        # Combine
        root = Product(inputs=[simple_sum, complex_sum])

        # Build graph to verify structure
        graph = pydot.Dot(graph_type="digraph")
        _build_graph(root, graph)
        nodes = [n for n in graph.get_nodes() if n.get_name() not in ("node", "edge", "graph")]

        # Should have Product, Sum, Normal, Sum, Product, Normal, Categorical = 7 nodes
        assert len(nodes) == 7


class TestGraphvizErrorHandling:
    """Test error handling when Graphviz is not available."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        return Sum(inputs=leaf, out_channels=3)

    def test_missing_graphviz_executable_raises_graphviz_error(self, simple_model):
        """Test that FileNotFoundError from missing graphviz is wrapped in GraphvizError."""
        with patch("pydot.Dot.write_png") as mock_write:
            # Simulate graphviz executable not found
            mock_write.side_effect = FileNotFoundError(errno.ENOENT, "dot not found in path")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "test_model")
                with pytest.raises(GraphvizError, match="Graphviz executable.*not found"):
                    visualize(simple_model, output_path, format="png")

    def test_graphviz_assertion_error_raises_graphviz_error(self, simple_model):
        """Test that AssertionError from graphviz failure is wrapped in GraphvizError."""
        with patch("pydot.Dot.write_png") as mock_write:
            # Simulate graphviz returning non-zero exit code
            mock_write.side_effect = AssertionError('"dot" with args [...] returned code: 1')

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "test_model")
                with pytest.raises(GraphvizError, match="Error executing Graphviz"):
                    visualize(simple_model, output_path, format="png")

    def test_graphviz_oserror_raises_graphviz_error(self, simple_model):
        """Test that OSError from graphviz execution is wrapped in GraphvizError."""
        with patch("pydot.Dot.write_pdf") as mock_write:
            # Simulate OSError during graphviz execution
            mock_write.side_effect = OSError("Error executing graphviz")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "test_model")
                with pytest.raises(GraphvizError, match="Error executing Graphviz"):
                    visualize(simple_model, output_path, format="pdf")
