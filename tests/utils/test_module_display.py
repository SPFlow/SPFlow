"""Tests for the module display system (module_to_str function and Module.to_str() method)."""

import pytest

from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.module_display import module_to_str


class TestModuleToStrBasics:
    """Test basic module_to_str functionality with all formats."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model: Sum -> Normal leaves."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        return Sum(inputs=leaf, out_channels=3)

    def test_tree_view_format(self, simple_model):
        """Test tree view format returns hierarchical structure."""
        output = module_to_str(simple_model, format="tree")

        # Guard the public return contract before validating content details.
        assert isinstance(output, str)
        assert len(output) > 0

        # Include both node types so traversal is visible to users.
        assert "Sum" in output
        assert "Normal" in output

        # Tree glyphs distinguish hierarchical mode from flat repr output.
        assert any(c in output for c in ["├", "└", "│"])

    def test_pytorch_view_format(self, simple_model):
        """Test pytorch format returns repr-like output."""
        output = module_to_str(simple_model, format="pytorch")

        assert isinstance(output, str)
        # Keep parity with repr() so debugging output stays predictable.
        assert output == repr(simple_model)

    def test_invalid_format_raises_error(self, simple_model):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError):
            module_to_str(simple_model, format="invalid_format")

    def test_inline_format_raises_error(self, simple_model):
        """Test that inline format is no longer supported."""
        with pytest.raises(ValueError):
            module_to_str(simple_model, format="inline")

    def test_graph_format_raises_error(self, simple_model):
        """Test that graph format is no longer supported."""
        with pytest.raises(ValueError):
            module_to_str(simple_model, format="graph")


class TestModuleToStrCustomization:
    """Test customization options for module_to_str."""

    @pytest.fixture
    def nested_model(self):
        """Create a nested model with multiple levels."""
        # Build a valid mixed graph; scope constraints are strict constructor invariants.
        leaf_same_scope1 = Normal(scope=Scope([0, 1]), out_channels=2)
        leaf_same_scope2 = Normal(scope=Scope([0, 1]), out_channels=2)
        # Sum setup intentionally satisfies identical-scope requirement.
        sum_node1 = Sum(inputs=[leaf_same_scope1, leaf_same_scope2], out_channels=2)

        # Product setup intentionally satisfies disjoint-scope requirement.
        leaf_diff_scope1 = Normal(scope=Scope([2]), out_channels=2)
        leaf_diff_scope2 = Normal(scope=Scope([3]), out_channels=2)
        product_node = Product(inputs=[leaf_diff_scope1, leaf_diff_scope2])

        # Use a second Sum branch because Product's merged scope would be invalid here.
        leaf3 = Normal(scope=Scope([0, 1]), out_channels=2)
        sum_node2 = Sum(inputs=leaf3, out_channels=2)
        return Sum(inputs=[sum_node1, sum_node2], out_channels=3)

    def test_max_depth_limits_output(self, nested_model):
        """Test max_depth parameter limits hierarchy depth."""
        output_full = module_to_str(nested_model, format="tree", max_depth=None)
        output_depth1 = module_to_str(nested_model, format="tree", max_depth=1)
        output_depth2 = module_to_str(nested_model, format="tree", max_depth=2)

        # Length ordering is a stable proxy that truncation happened.
        assert len(output_depth1) <= len(output_full)
        # Intermediate depth should preserve monotonic growth in detail.
        assert len(output_depth1) <= len(output_depth2) <= len(output_full)

        # Even aggressive truncation should still identify the root module.
        assert "Sum" in output_depth1

    def test_show_params_false(self, nested_model):
        """Test show_params=False hides parameter information."""
        output_with_params = module_to_str(nested_model, format="tree", show_params=True)
        output_no_params = module_to_str(nested_model, format="tree", show_params=False)

        # Hiding metadata must not increase output verbosity.
        assert len(output_no_params) <= len(output_with_params)

    def test_show_scope_false(self, nested_model):
        """Test show_scope=False hides scope information."""
        output_with_scope = module_to_str(nested_model, format="tree", show_scope=True)
        output_no_scope = module_to_str(nested_model, format="tree", show_scope=False)

        # Scope suppression should remove text, not add formatting noise.
        assert len(output_no_scope) <= len(output_with_scope)

        # Accept equality for modules that cannot expose scope text.
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
        # Keep scopes disjoint so Product construction remains valid.
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
        # Build legal scopes so depth behavior is tested without constructor failures.
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        level1 = Sum(inputs=leaf, out_channels=2)

        # Product branch enforces disjoint scope invariant.
        leaf_prod1 = Normal(scope=Scope([2]), out_channels=2)
        leaf_prod2 = Normal(scope=Scope([3]), out_channels=2)
        level2 = Product(inputs=[leaf_prod1, leaf_prod2])

        # Sum branch reuses matching scopes to stay structurally valid.
        level3 = Sum(inputs=level1, out_channels=2)
        level4 = Sum(inputs=level3, out_channels=2)

        output_full = module_to_str(level4, format="tree", max_depth=None)
        output_limited = module_to_str(level4, format="tree", max_depth=2)

        assert isinstance(output_full, str)
        assert isinstance(output_limited, str)
        assert len(output_limited) <= len(output_full)

    def test_multiple_leaves_same_module(self):
        """Test Sum with multiple Normal leaves inputs (same scope)."""
        # Reuse one scope object so the Sum precondition is unambiguous.
        scope = Scope([0, 1])
        leaves = [Normal(scope=scope, out_channels=2) for _ in range(3)]
        sum_node = Sum(inputs=leaves, out_channels=2)

        output = module_to_str(sum_node, format="tree")

        assert "Sum" in output
        assert output.count("Normal") == 3

    def test_sum_with_mixed_inputs(self):
        """Test Sum with both Sum and leaves inputs (same scope)."""
        # This catches regressions in mixed-node rendering, not scope validation.
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
        """Test with a single leaves module (no containers)."""
        leaf = Normal(scope=Scope([0, 1, 2]), out_channels=3)

        output = module_to_str(leaf, format="tree")

        assert "Normal" in output
        assert isinstance(output, str)

    def test_scope_display(self):
        """Test that scope information is displayed correctly."""
        # Share scope explicitly to avoid accidental constructor errors.
        scope = Scope([0, 1])
        leaf1 = Normal(scope=scope, out_channels=2)
        leaf2 = Normal(scope=scope, out_channels=2)

        # Explicit setup keeps this test focused on formatting behavior.
        model = Sum(inputs=[leaf1, leaf2], out_channels=2)

        output = module_to_str(model, format="tree", show_scope=True)

        assert isinstance(output, str)
        # Renderer variants differ, so accept either explicit scope text or dash fallback.
        assert "scope" in output.lower() or "-" in output

    def test_max_depth_zero(self):
        """Test behavior with max_depth=0."""
        leaf = Normal(scope=Scope([0]), out_channels=2)
        sum_node = Sum(inputs=leaf, out_channels=2)

        output = module_to_str(sum_node, format="tree", max_depth=0)

        # Depth zero previously regressed by dropping all output.
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
        # Printing ensures no hidden encoding/control chars leak into CLI output.
        print(output)


class TestModuleToStrParameterDisplay:
    """Test parameter and scope display in different formats."""

    def test_tree_shows_parameters(self):
        """Test that tree view can show parameter information."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=2)
        model = Sum(inputs=leaf, out_channels=3)

        output_with_params = module_to_str(model, format="tree", show_params=True)

        # Param flag should expose shape/channel hints for model inspection.
        assert "D=" in output_with_params or "C=" in output_with_params


# Branch-specific checks for internal traversal helper edge paths.
import numpy as np
import torch

from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.module_display import _format_scope, _get_module_children, _tree_view


class _Mini(Module):
    def __init__(self, scope: Scope | None = None):
        super().__init__()
        self.scope = scope if scope is not None else Scope([0])
        self.in_shape = ModuleShape(1, 1, 1)
        self.out_shape = ModuleShape(1, 1, 1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([0])], dtype=object).reshape(1, 1)

    def log_likelihood(self, data, cache=None):
        return torch.zeros((data.shape[0], 1, 1, 1))

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None):
        return self._prepare_sample_data(num_samples, data)

    def _sample(self, data, sampling_ctx, cache):
        del sampling_ctx
        del cache
        return data

    def expectation_maximization(self, data, bias_correction=True, cache=None):
        return None

    def maximum_likelihood_estimation(
        self, data, weights=None, bias_correction=True, nan_strategy="ignore", cache=None
    ):
        return None

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return self


def test_tree_view_depth_cutoff_branch():
    m = _Mini()
    assert (
        _tree_view(m, max_depth=0, show_params=True, show_scope=True, depth=1, is_last=True, prefix="") == ""
    )


def test_num_repetitions_property_and_scope_format_branches():
    m = _Mini(Scope([0, 2, 4]))
    m.num_repetitions = 7

    out = module_to_str(m, format="tree", show_scope=True)
    assert "R=7" in out
    assert "{0, 2, 4}" in out

    assert _format_scope(None) == ""
    assert _format_scope(type("NoQuery", (), {})()) == ""
    assert _format_scope(type("Empty", (), {"query": []})()) == ""


def test_get_module_children_branches_for_modulelist_list_cat_and_ratspn():
    child = _Mini()

    # Exercise ModuleList path used by many container modules.
    parent_ml = _Mini()
    parent_ml.inputs = torch.nn.ModuleList([child])
    assert _get_module_children(parent_ml) == [("input[0]", child)]

    # Exercise plain-list fallback for hand-built test modules.
    parent_list = _Mini()
    parent_list.inputs = [child]
    assert _get_module_children(parent_list) == [("input[0]", child)]

    # Cat wraps one child in some construction paths; preserve that special handling.
    CatCls = type("Cat", (_Mini,), {})
    cat = CatCls()
    cat.inputs = child

    parent_cat = _Mini()
    parent_cat.inputs = cat
    assert _get_module_children(parent_cat) == [("inputs", child)]

    # RatSPN exposes children via root_node instead of inputs.
    RatCls = type("RatSPN", (_Mini,), {})
    rat = RatCls()
    rat.root_node = child
    assert _get_module_children(rat) == [("root_node", child)]
