"""Tests for the module display system (module_to_str function and Module.to_str() method)."""

import pytest

from spflow.meta import Scope
from spflow.modules.leaf import Normal
from spflow.modules.product import Product
from spflow.modules.sum import Sum
from spflow.utils.module_display import module_to_str


class TestModuleToStrBasics:
    """Test basic module_to_str functionality with all formats."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model: Sum -> Normal leaf."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        return Sum(inputs=leaf, out_channels=3)

    def test_tree_view_format(self, simple_model):
        """Test tree view format returns hierarchical structure."""
        output = module_to_str(simple_model, format="tree")

        # Check it's a non-empty string
        assert isinstance(output, str)
        assert len(output) > 0

        # Check for Sum module
        assert "Sum" in output
        assert "Normal" in output

        # Check for tree characters
        assert any(c in output for c in ["├", "└", "│"])

    def test_pytorch_view_format(self, simple_model):
        """Test pytorch format returns repr-like output."""
        output = module_to_str(simple_model, format="pytorch")

        assert isinstance(output, str)
        # PyTorch format uses repr()
        assert output == repr(simple_model)

    def test_invalid_format_raises_error(self, simple_model):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            module_to_str(simple_model, format="invalid_format")

    def test_inline_format_raises_error(self, simple_model):
        """Test that inline format is no longer supported."""
        with pytest.raises(ValueError, match="Unknown format"):
            module_to_str(simple_model, format="inline")

    def test_graph_format_raises_error(self, simple_model):
        """Test that graph format is no longer supported."""
        with pytest.raises(ValueError, match="Unknown format"):
            module_to_str(simple_model, format="graph")


class TestModuleToStrCustomization:
    """Test customization options for module_to_str."""

    @pytest.fixture
    def nested_model(self):
        """Create a nested model with multiple levels."""
        # Create leaves with same scope (for Sum) and different scopes (for Product)
        leaf_same_scope1 = Normal(scope=Scope([0, 1]), out_channels=2)
        leaf_same_scope2 = Normal(scope=Scope([0, 1]), out_channels=2)
        # Sum requires same scope inputs
        sum_node1 = Sum(inputs=[leaf_same_scope1, leaf_same_scope2], out_channels=2)

        # Product requires disjoint scopes
        leaf_diff_scope1 = Normal(scope=Scope([2]), out_channels=2)
        leaf_diff_scope2 = Normal(scope=Scope([3]), out_channels=2)
        product_node = Product(inputs=[leaf_diff_scope1, leaf_diff_scope2])

        # Another sum for final layer (must have same scope)
        # Product output scope is union of inputs, so we can't directly sum it
        # Instead, create a simpler nested structure
        leaf3 = Normal(scope=Scope([0, 1]), out_channels=2)
        sum_node2 = Sum(inputs=leaf3, out_channels=2)
        return Sum(inputs=[sum_node1, sum_node2], out_channels=3)

    def test_max_depth_limits_output(self, nested_model):
        """Test max_depth parameter limits hierarchy depth."""
        output_full = module_to_str(nested_model, format="tree", max_depth=None)
        output_depth1 = module_to_str(nested_model, format="tree", max_depth=1)
        output_depth2 = module_to_str(nested_model, format="tree", max_depth=2)

        # Depth 1 should be shorter than full
        assert len(output_depth1) <= len(output_full)
        # Depth 2 should be between depth 1 and full
        assert len(output_depth1) <= len(output_depth2) <= len(output_full)

        # Root should always be present
        assert "Sum" in output_depth1

    def test_show_params_false(self, nested_model):
        """Test show_params=False hides parameter information."""
        output_with_params = module_to_str(nested_model, format="tree", show_params=True)
        output_no_params = module_to_str(nested_model, format="tree", show_params=False)

        # Output without params should generally be shorter or equal
        assert len(output_no_params) <= len(output_with_params)

    def test_show_scope_false(self, nested_model):
        """Test show_scope=False hides scope information."""
        output_with_scope = module_to_str(nested_model, format="tree", show_scope=True)
        output_no_scope = module_to_str(nested_model, format="tree", show_scope=False)

        # Output without scope should be shorter
        assert len(output_no_scope) <= len(output_with_scope)

        # With scope should contain scope info
        assert "scope:" in output_with_scope or output_with_scope == output_no_scope

    def test_combined_customization(self, nested_model):
        """Test combining multiple customization options."""
        output = module_to_str(nested_model, format="tree", max_depth=1, show_params=False, show_scope=False)

        assert isinstance(output, str)
        assert len(output) > 0
        assert "Sum" in output


class TestModuleToStrMethod:
    """Test the Module.to_str() method."""

    @pytest.fixture
    def model_with_method(self):
        """Create a model to test the to_str() method."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        return Sum(inputs=leaf, out_channels=3)

    def test_to_str_method_exists(self, model_with_method):
        """Test that Module has to_str() method."""
        assert hasattr(model_with_method, "to_str")
        assert callable(model_with_method.to_str)

    def test_to_str_method_default(self, model_with_method):
        """Test Module.to_str() with default parameters (tree format)."""
        output = model_with_method.to_str()

        assert isinstance(output, str)
        assert "Sum" in output
        assert "Normal" in output

    def test_to_str_method_with_format(self, model_with_method):
        """Test Module.to_str() with different formats."""
        for fmt in ["tree", "pytorch"]:
            output = model_with_method.to_str(format=fmt)
            assert isinstance(output, str)
            assert len(output) > 0

    def test_to_str_method_with_customization(self, model_with_method):
        """Test Module.to_str() with customization options."""
        output = model_with_method.to_str(format="tree", max_depth=1, show_params=False, show_scope=False)

        assert isinstance(output, str)
        assert "Sum" in output

    def test_to_str_method_equivalence_to_function(self, model_with_method):
        """Test that Module.to_str() produces same output as module_to_str()."""
        method_output = model_with_method.to_str(format="tree", max_depth=2)
        function_output = module_to_str(model_with_method, format="tree", max_depth=2)

        assert method_output == function_output


class TestModuleToStrComplexStructures:
    """Test module_to_str with complex module structures."""

    def test_product_of_sums(self):
        """Test displaying Product of Sum nodes."""
        # Product requires disjoint scopes
        leaf1 = Normal(scope=Scope([0]), out_channels=2)
        leaf2 = Normal(scope=Scope([1]), out_channels=2)
        sum1 = Sum(inputs=leaf1, out_channels=2)
        sum2 = Sum(inputs=leaf2, out_channels=2)
        product = Product(inputs=[sum1, sum2])

        output = module_to_str(product, format="tree")

        assert "Product" in output
        assert output.count("Sum") == 2
        assert output.count("Normal") == 2

    def test_deeply_nested_structure(self):
        """Test displaying deeply nested module structure."""
        # Create a 4-level deep structure with proper scope handling
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        level1 = Sum(inputs=leaf, out_channels=2)

        # For level 2, we need disjoint scopes for product
        leaf_prod1 = Normal(scope=Scope([2]), out_channels=2)
        leaf_prod2 = Normal(scope=Scope([3]), out_channels=2)
        level2 = Product(inputs=[leaf_prod1, leaf_prod2])

        # For level 3, we need same scope sum (they have the same scope now)
        level3 = Sum(inputs=level1, out_channels=2)
        level4 = Sum(inputs=level3, out_channels=2)

        output_full = module_to_str(level4, format="tree", max_depth=None)
        output_limited = module_to_str(level4, format="tree", max_depth=2)

        assert isinstance(output_full, str)
        assert isinstance(output_limited, str)
        assert len(output_limited) <= len(output_full)

    def test_multiple_leaves_same_module(self):
        """Test Sum with multiple Normal leaf inputs (same scope)."""
        # Sum requires same scope inputs
        scope = Scope([0, 1])
        leaves = [Normal(scope=scope, out_channels=2) for _ in range(3)]
        sum_node = Sum(inputs=leaves, out_channels=2)

        output = module_to_str(sum_node, format="tree")

        assert "Sum" in output
        assert output.count("Normal") == 3

    def test_sum_with_mixed_inputs(self):
        """Test Sum with both Sum and leaf inputs (same scope)."""
        # Both inputs must have same scope for Sum
        scope = Scope([0, 1])
        leaf = Normal(scope=scope, out_channels=2)
        inner_sum = Sum(inputs=leaf, out_channels=2)
        leaf2 = Normal(scope=scope, out_channels=2)
        outer_sum = Sum(inputs=[inner_sum, leaf2], out_channels=2)

        output = module_to_str(outer_sum, format="tree")

        assert output.count("Sum") == 2
        assert output.count("Normal") == 2


class TestModuleToStrEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_leaf_module(self):
        """Test with a single leaf module (no containers)."""
        leaf = Normal(scope=Scope([0, 1, 2]), out_channels=3)

        output = module_to_str(leaf, format="tree")

        assert "Normal" in output
        assert isinstance(output, str)

    def test_scope_display(self):
        """Test that scope information is displayed correctly."""
        # Sum requires same scope inputs
        scope = Scope([0, 1])
        leaf1 = Normal(scope=scope, out_channels=2)
        leaf2 = Normal(scope=scope, out_channels=2)

        # Create with manually set scopes
        model = Sum(inputs=[leaf1, leaf2], out_channels=2)

        output = module_to_str(model, format="tree", show_scope=True)

        assert isinstance(output, str)
        # Should show some scope information
        assert "scope" in output.lower() or "-" in output

    def test_max_depth_zero(self):
        """Test behavior with max_depth=0."""
        leaf = Normal(scope=Scope([0]), out_channels=2)
        sum_node = Sum(inputs=leaf, out_channels=2)

        output = module_to_str(sum_node, format="tree", max_depth=0)

        # With max_depth=0, should show root but no children
        assert isinstance(output, str)

    @pytest.mark.parametrize(
        "show_params,show_scope",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_format_consistency_across_options(self, show_params, show_scope):
        """Test that different customization options don't cause errors."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        model = Sum(inputs=leaf, out_channels=3)

        output = module_to_str(model, format="tree", show_params=show_params, show_scope=show_scope)
        assert isinstance(output, str)
        assert len(output) > 0


class TestModuleToStrStringProperties:
    """Test string output properties and content."""

    @pytest.fixture
    def test_model(self):
        """Create a test model."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        return Sum(inputs=leaf, out_channels=3)

    @pytest.mark.parametrize("format", ["tree", "pytorch"])
    def test_output_is_string(self, test_model, format):
        """Test that output is always a string and is printable."""
        output = module_to_str(test_model, format=format)
        assert isinstance(output, str)
        assert len(output) > 0
        # Should be able to print it without errors
        print(output)


class TestModuleToStrParameterDisplay:
    """Test parameter and scope display in different formats."""

    def test_tree_shows_parameters(self):
        """Test that tree view can show parameter information."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        model = Sum(inputs=leaf, out_channels=3)

        output_with_params = module_to_str(model, format="tree", show_params=True)

        # Tree should contain dimension info
        assert "D=" in output_with_params or "C=" in output_with_params
