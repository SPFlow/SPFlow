"""Tests for structure statistics utility."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spflow.exceptions import StructureError
from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.leaves import Normal
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.structure_stats import (
    StructureStats,
    _iter_structure_children,
    _tensor_identity,
    get_structure_stats,
    structure_stats_to_str,
)


class _DummyModule(Module):
    def __init__(self, *, inputs=None, wrapped=None, root_node=None, scope=None, params=None):
        super().__init__()
        if inputs is not None:
            self.inputs = inputs
        if wrapped is not None:
            self.module = wrapped
        if root_node is not None:
            self.root_node = root_node
        if scope is not None:
            self.scope = scope
        if params:
            for name, param in params.items():
                self.register_parameter(name, param)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.empty((0,), dtype=object)

    def log_likelihood(self, data, cache=None):
        return torch.zeros(1)

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None):
        return torch.zeros(1)

    def _sample(self, data, sampling_ctx, cache):
        del data
        del sampling_ctx
        del cache
        return torch.zeros(1)

    def marginalize(self, scope):
        return self


class _CatDummy(_DummyModule):
    pass


_CatDummy.__name__ = "Cat"


class _RatSPNDummy(_DummyModule):
    pass


_RatSPNDummy.__name__ = "RatSPN"


class _BadScope:
    def __len__(self) -> int:
        raise RuntimeError("scope length unavailable")


def test_stats_counts_on_known_small_model() -> None:
    leaf0 = Normal(scope=Scope([0]), out_channels=1)
    leaf1 = Normal(scope=Scope([1]), out_channels=1)
    cat = Cat(inputs=[leaf0, leaf1], dim=1)
    prod = Product(inputs=cat)
    model = Sum(inputs=prod, out_channels=2)

    stats = get_structure_stats(model)
    assert isinstance(stats, StructureStats)

    assert stats.num_nodes_total == 4  # Sum, Product, Normal, Normal (Cat is skipped)
    assert stats.num_edges_total == 3  # Sum->Product, Product->(leaf0, leaf1)
    assert stats.max_depth == 3  # Sum->Product->Leaf (leaf depth=1)

    node_type_counts = stats.node_type_counts
    assert node_type_counts["Sum"] == 1
    assert node_type_counts["Product"] == 1
    assert node_type_counts["Normal"] == 2


def test_parameter_count_matches_sum_of_unique_parameters() -> None:
    shared_leaf = Normal(scope=Scope([0]), out_channels=1)
    model = Sum(inputs=[shared_leaf, shared_leaf], out_channels=1)

    stats = get_structure_stats(model)

    # The leaf is shared twice but should be counted once as a node.
    assert stats.num_nodes_total == 2

    # Edges count both incoming references to the shared leaf.
    assert stats.num_edges_total == 2
    assert stats.num_shared_nodes == 1
    assert stats.is_dag is True

    # Parameters should be counted uniquely across shared subgraphs.
    expected = sum(int(p.numel()) for p in model.parameters())
    assert stats.num_parameters_total == expected


def test_depth_computation() -> None:
    leaf0 = Normal(scope=Scope([0]), out_channels=1)
    leaf1 = Normal(scope=Scope([1]), out_channels=1)
    prod = Product(inputs=Cat(inputs=[leaf0, leaf1], dim=1))
    sum1 = Sum(inputs=prod, out_channels=2)
    sum2 = Sum(inputs=sum1, out_channels=1)

    stats = get_structure_stats(sum2)
    assert stats.max_depth == 4  # Sum2->Sum1->Product->Leaf


def test_dag_shared_nodes_count() -> None:
    shared_leaf = Normal(scope=Scope([0]), out_channels=1)
    model = Sum(inputs=[shared_leaf, shared_leaf], out_channels=2)

    stats = get_structure_stats(model)
    assert stats.num_shared_nodes == 1


def test_print_structure_stats_method() -> None:
    leaf0 = Normal(scope=Scope([0]), out_channels=1)
    leaf1 = Normal(scope=Scope([1]), out_channels=1)
    model = Sum(inputs=Product(inputs=Cat(inputs=[leaf0, leaf1], dim=1)), out_channels=2)

    text = model.print_structure_stats()
    assert "Structure statistics" in text
    assert "nodes:" in text
    assert "edges:" in text
    assert "parameters:" in text


def test_structure_stats_dataclass_to_dict() -> None:
    stats = StructureStats(
        num_nodes_total=1,
        num_edges_total=0,
        num_parameters_total=3,
        node_type_counts={"A": 1},
        parameter_type_counts={"weight": 1},
        max_depth=1,
        scope_size_min=1,
        scope_size_max=1,
        scope_size_mean=1.0,
        scope_size_histogram={1: 1},
        is_dag=False,
        num_shared_nodes=0,
    )
    as_dict = stats.to_dict()
    assert as_dict["num_nodes_total"] == 1
    assert as_dict["scope_size_histogram"] == {1: 1}


def test_tensor_identity_fallback_branch() -> None:
    key = _tensor_identity(object())
    assert key[0] == "id"
    assert len(key) == 2


def test_iter_structure_children_handles_wrapper_list_modulelist_cat_and_ratspn() -> None:
    leaf0 = _DummyModule()
    leaf1 = _DummyModule()
    leaf2 = _DummyModule()

    wrapped_parent = _DummyModule(wrapped=leaf0)
    assert _iter_structure_children(wrapped_parent) == [leaf0]

    list_parent = _DummyModule(inputs=[leaf0, object(), leaf1])
    assert _iter_structure_children(list_parent) == [leaf0, leaf1]

    modulelist_parent = _DummyModule(inputs=torch.nn.ModuleList([leaf0, leaf1]))
    assert _iter_structure_children(modulelist_parent) == [leaf0, leaf1]

    cat_module = _CatDummy(inputs=leaf2)
    cat_parent = _DummyModule(inputs=cat_module)
    assert _iter_structure_children(cat_parent) == [leaf2]

    ratspn = _RatSPNDummy(root_node=leaf0)
    assert _iter_structure_children(ratspn) == [leaf0]


def test_scope_len_exception_and_none_parameter_are_ignored() -> None:
    class _NoneParamNode(_DummyModule):
        def named_parameters(self, prefix="", recurse=False, remove_duplicate=True):
            yield "none_param", None

    child = _NoneParamNode(scope=_BadScope())
    root = _DummyModule(inputs=child)
    stats = get_structure_stats(root)
    assert stats.num_nodes_total == 2
    assert stats.num_parameters_total == 0


def test_structure_stats_cycle_detection_raises() -> None:
    a = _DummyModule()
    b = _DummyModule(inputs=a)
    a.inputs = b

    with pytest.raises(StructureError):
        get_structure_stats(a)


def test_structure_stats_to_str_shows_node_types_more() -> None:
    stats = StructureStats(
        num_nodes_total=5,
        num_edges_total=4,
        num_parameters_total=0,
        node_type_counts={"A": 1, "B": 1, "C": 1},
        parameter_type_counts={},
        max_depth=2,
        scope_size_min=None,
        scope_size_max=None,
        scope_size_mean=None,
        scope_size_histogram={},
        is_dag=False,
        num_shared_nodes=0,
    )
    text = structure_stats_to_str(stats, max_node_types=2)
    assert "node_types_more: 1" in text
