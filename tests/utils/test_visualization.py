"""Tests for the graph visualization utilities."""

import os
import shutil
import tempfile

import pytest
import pydot

from spflow.meta import Scope
from spflow.modules.leaf import Categorical, Normal
from spflow.modules.product import Product
from spflow.modules.sum import Sum
from spflow.utils.visualization import (
    visualize_module,
    _build_graph,
    _format_scope_string,
    _format_param_count,
    _count_parameters,
)


def has_graphviz_dot():
    """Check if the graphviz 'dot' binary is available on the system."""
    return shutil.which("dot") is not None


# Skip all visualization tests that require graphviz if 'dot' binary is not available
pytestmark_requires_graphviz = pytest.mark.skipif(
    not has_graphviz_dot(),
    reason="graphviz 'dot' binary not found. Install graphviz to run visualization tests.",
)


class TestFormatScopeString:
    """Test the _format_scope_string function for formatting scope indices."""

    def test_empty_list(self):
        """Test formatting an empty list of scopes."""
        assert _format_scope_string([]) == ""

    def test_single_element(self):
        """Test formatting a single scope."""
        assert _format_scope_string([5]) == "5"

    def test_two_elements(self):
        """Test formatting two non-consecutive scopes."""
        assert _format_scope_string([5, 10]) == "5, 10"

    def test_two_consecutive_elements(self):
        """Test formatting two consecutive scopes."""
        assert _format_scope_string([5, 6]) == "5, 6"

    def test_three_consecutive_elements(self):
        """Test formatting three consecutive scopes (should use range with 3+ threshold)."""
        assert _format_scope_string([5, 6, 7]) == "5...7"

    def test_four_consecutive_elements(self):
        """Test formatting four consecutive scopes (should use range)."""
        assert _format_scope_string([5, 6, 7, 8]) == "5...8"

    def test_complex_mix(self):
        """Test formatting with mixed consecutive and non-consecutive scopes.

        This matches the example in the docstring:
        [0,1,2,3,4,8,9,10,23,24,25,26,90] -> "0...4, 8...10, 23...26, 90"
        """
        scopes = [0, 1, 2, 3, 4, 8, 9, 10, 23, 24, 25, 26, 90]
        assert _format_scope_string(scopes) == "0...4, 8...10, 23...26, 90"

    def test_unsorted_input(self):
        """Test that unsorted input is handled correctly."""
        assert _format_scope_string([5, 1, 2, 3, 10]) == "1...3, 5, 10"

    def test_duplicate_elements(self):
        """Test that duplicate elements are handled correctly."""
        assert _format_scope_string([1, 2, 2, 3, 3, 3]) == "1...3"

    def test_single_large_range(self):
        """Test formatting a single large consecutive range."""
        assert _format_scope_string(list(range(0, 10))) == "0...9"

    def test_multiple_ranges_and_singles(self):
        """Test multiple ranges with single elements in between."""
        scopes = [0, 1, 2, 3, 10, 15, 16, 17, 18, 19, 25]
        assert _format_scope_string(scopes) == "0...3, 10, 15...19, 25"

    def test_ranges_at_boundaries(self):
        """Test ranges that start at 0 and include high numbers."""
        assert _format_scope_string([0, 1, 2, 3, 100, 101, 102, 103]) == "0...3, 100...103"


class TestFormatParamCount:
    """Test the _format_param_count function for formatting parameter counts."""

    def test_format_small_count(self):
        """Test formatting parameter counts less than 1000."""
        assert _format_param_count(0) == "0"
        assert _format_param_count(1) == "1"
        assert _format_param_count(42) == "42"
        assert _format_param_count(999) == "999"

    def test_format_thousands(self):
        """Test formatting parameter counts in thousands."""
        assert _format_param_count(1000) == "1.0K"
        assert _format_param_count(1234) == "1.2K"
        assert _format_param_count(5678) == "5.7K"
        assert _format_param_count(999_999) == "1000.0K"

    def test_format_millions(self):
        """Test formatting parameter counts in millions."""
        assert _format_param_count(1_000_000) == "1.0M"
        assert _format_param_count(1_234_567) == "1.2M"
        assert _format_param_count(10_000_000) == "10.0M"

    def test_count_parameters_leaf_module(self):
        """Test counting parameters in a leaf module."""
        # Normal module stores its parameters (mean, std) in a child Distribution module
        # Leaf modules count all parameters including the distribution child
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        param_count = _count_parameters(leaf)
        # Should have parameters (mean and std for 2 features x 2 channels)
        assert param_count > 0

    def test_count_parameters_sum_module(self):
        """Test counting parameters in a sum module."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        sum_module = Sum(inputs=leaf, out_channels=3)
        param_count = _count_parameters(sum_module)
        # Should have parameters (only the sum module's own weights, not leaf parameters)
        assert param_count > 0


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
        nodes = [n for n in nodes if n.get_name() not in ('node', 'edge', 'graph')]

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
        nodes = [n for n in nodes if n.get_name() not in ('node', 'edge', 'graph')]

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
        """Create a simple model: Sum -> Normal leaf."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        return Sum(inputs=leaf, out_channels=3)

    @pytest.mark.parametrize("format", ["png", "pdf", "svg"])
    def test_visualize_formats(self, simple_model, format):
        """Test that visualize_module creates output files in different formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_model")
            visualize_module(simple_model, output_path, format=format)
            assert os.path.exists(f"{output_path}.{format}")

    def test_invalid_format_raises_error(self, simple_model):
        """Test that invalid format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_model")
            with pytest.raises(ValueError, match="Unsupported format"):
                visualize_module(simple_model, output_path, format="invalid_format")


class TestVisualizationOptions:
    """Test visualization customization options."""

    pytestmark = pytestmark_requires_graphviz

    @pytest.fixture
    def model_with_scope(self):
        """Create a model with specific scopes."""
        leaf1 = Normal(scope=Scope([0, 1]), out_channels=2)
        leaf2 = Normal(scope=Scope([2, 3]), out_channels=2)
        return Product(inputs=[leaf1, leaf2])

    def test_visualize_with_all_options_enabled(self, model_with_scope):
        """Test visualization with all customization options enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_with_all_options")
            visualize_module(
                model_with_scope,
                output_path,
                show_scope=True,
                show_shape=True,
                show_params=True,
                dpi=150,
                format="png"
            )
            assert os.path.exists(f"{output_path}.png")

    def test_visualize_with_all_options_disabled(self, model_with_scope):
        """Test visualization with all customization options disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_with_no_options")
            visualize_module(
                model_with_scope,
                output_path,
                show_scope=False,
                show_shape=False,
                show_params=False,
                format="png"
            )
            assert os.path.exists(f"{output_path}.png")

    @pytest.mark.parametrize("show_scope,show_shape,show_params", [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ])
    def test_visualize_individual_options(self, model_with_scope, show_scope, show_shape, show_params):
        """Test visualization with individual customization options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, f"test_options_{show_scope}_{show_shape}_{show_params}")
            visualize_module(
                model_with_scope,
                output_path,
                show_scope=show_scope,
                show_shape=show_shape,
                show_params=show_params,
                format="png"
            )
            assert os.path.exists(f"{output_path}.png")


class TestVisualizationEngines:
    """Test different graphviz layout engines."""

    pytestmark = pytestmark_requires_graphviz

    @pytest.fixture
    def nested_model(self):
        """Create a nested model for layout testing."""
        leaf1 = Normal(scope=Scope([0]), out_channels=2)
        leaf2 = Normal(scope=Scope([1]), out_channels=2)
        sum1 = Sum(inputs=leaf1, out_channels=2)
        sum2 = Sum(inputs=leaf2, out_channels=2)
        return Product(inputs=[sum1, sum2])

    @pytest.mark.parametrize("engine", ["dot", "dot-lr", "neato", "circo", "fdp"])
    def test_visualization_engines(self, nested_model, engine):
        """Test different graphviz layout engines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, f"test_{engine}")
            visualize_module(nested_model, output_path, engine=engine, format="png")
            assert os.path.exists(f"{output_path}.png")


class TestVisualizationRankdir:
    """Test different rankdir options."""

    pytestmark = pytestmark_requires_graphviz

    @pytest.fixture
    def simple_chain(self):
        """Create a simple chain model."""
        leaf = Normal(scope=Scope([0]), out_channels=2)
        sum1 = Sum(inputs=leaf, out_channels=2)
        sum2 = Sum(inputs=sum1, out_channels=2)
        return sum2

    @pytest.mark.parametrize("rankdir", ["TB", "LR", "BT", "RL"])
    def test_rankdir_options(self, simple_chain, rankdir):
        """Test different graph direction options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, f"test_rankdir_{rankdir}")
            visualize_module(simple_chain, output_path, rankdir=rankdir, format="png")
            assert os.path.exists(f"{output_path}.png")


class TestVisualizationComplexStructures:
    """Test visualization with complex module structures."""

    pytestmark = pytestmark_requires_graphviz

    def test_product_of_sums(self):
        """Test visualizing Product of Sum nodes."""
        leaf1 = Normal(scope=Scope([0]), out_channels=2)
        leaf2 = Normal(scope=Scope([1]), out_channels=2)
        sum1 = Sum(inputs=leaf1, out_channels=2)
        sum2 = Sum(inputs=leaf2, out_channels=2)
        product = Product(inputs=[sum1, sum2])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_product_of_sums")
            visualize_module(product, output_path, format="png")
            assert os.path.exists(f"{output_path}.png")

    def test_deeply_nested_structure(self):
        """Test visualizing deeply nested module structure with mixed types."""
        leaf = Normal(scope=Scope([0]), out_channels=2)
        level1 = Sum(inputs=leaf, out_channels=2)

        # Add Product level for more complexity
        leaf2 = Normal(scope=Scope([1]), out_channels=2)
        product = Product(inputs=[level1, leaf2])
        level3 = Sum(inputs=product, out_channels=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_deep_nested")
            visualize_module(level3, output_path, format="png")
            assert os.path.exists(f"{output_path}.png")


class TestVisualizationEdgeCases:
    """Test edge cases and special scenarios."""

    pytestmark = pytestmark_requires_graphviz

    def test_single_leaf_module(self):
        """Test visualization with a single leaf module."""
        leaf = Normal(scope=Scope([0, 1, 2]), out_channels=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_single_leaf")
            visualize_module(leaf, output_path, format="png")
            assert os.path.exists(f"{output_path}.png")

    def test_large_scope(self):
        """Test visualization with large scope."""
        leaf = Normal(scope=Scope(list(range(10))), out_channels=2)
        model = Sum(inputs=leaf, out_channels=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_large_scope")
            visualize_module(model, output_path, show_scope=True, format="png")
            assert os.path.exists(f"{output_path}.png")


class TestVisualizationIntegration:
    """Integration tests with realistic models."""

    pytestmark = pytestmark_requires_graphviz

    def test_typical_spn_structure(self):
        """Test visualization with a typical SPN structure."""
        # Create a small but realistic SPN
        # Leaf layer with disjoint scopes for Product
        leaf1 = Normal(scope=Scope([0]), out_channels=3)
        leaf2 = Normal(scope=Scope([1]), out_channels=3)
        leaf3 = Normal(scope=Scope([2]), out_channels=3)
        leaf4 = Normal(scope=Scope([3]), out_channels=3)

        # Product layer (combining disjoint scopes)
        prod1 = Product(inputs=[leaf1, leaf2])  # scope [0, 1]
        prod2 = Product(inputs=[leaf3, leaf4])  # scope [2, 3]

        # Sum layer (same scopes)
        sum1 = Sum(inputs=prod1, out_channels=2)  # scope [0, 1]
        sum2 = Sum(inputs=prod2, out_channels=2)  # scope [2, 3]

        # Root product (disjoint scopes)
        root_prod = Product(inputs=[sum1, sum2])  # scope [0, 1, 2, 3]

        # Root sum
        root = Sum(inputs=root_prod, out_channels=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_typical_spn")
            visualize_module(root, output_path, show_scope=True, format="png")
            assert os.path.exists(f"{output_path}.png")

    def test_mixed_complexity(self):
        """Test visualization with mixed complexity model."""
        # Simple branch
        simple_leaf = Normal(scope=Scope([0]), out_channels=2)
        simple_sum = Sum(inputs=simple_leaf, out_channels=2)

        # Complex branch
        complex_leaf1 = Normal(scope=Scope([1]), out_channels=3)
        complex_leaf2 = Categorical(scope=Scope([2]), out_channels=3, K=5)
        complex_prod = Product(inputs=[complex_leaf1, complex_leaf2])
        complex_sum = Sum(inputs=complex_prod, out_channels=2)

        # Combine (need compatible scopes)
        # Since simple_sum has scope [0] and complex_sum has scope [1, 2], we can use Product
        root = Product(inputs=[simple_sum, complex_sum])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_mixed_complexity")
            visualize_module(root, output_path, show_scope=True, show_shape=True, format="png")
            assert os.path.exists(f"{output_path}.png")
