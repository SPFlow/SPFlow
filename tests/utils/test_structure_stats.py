"""Tests for structure statistics utility."""

from __future__ import annotations

from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.structure_stats import StructureStats, get_structure_stats


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
