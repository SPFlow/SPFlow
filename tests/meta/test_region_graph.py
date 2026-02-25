import pytest

from spflow.meta.data.scope import Scope
from spflow.meta.region_graph import Region, RegionGraph


def test_add_partition_validates_disjoint_and_scope_match():
    root = Region(Scope([0, 1]))
    left = Region(Scope([0]))
    right = Region(Scope([1]))
    root.add_partition((left, right))
    assert len(root.children) == 1

    # Overlapping child scopes violate decomposability constraints.
    with pytest.raises(ValueError):
        root.add_partition((Region(Scope([0])), Region(Scope([0]))))

    # Child scopes must jointly cover the parent scope exactly.
    with pytest.raises(ValueError):
        root.add_partition((Region(Scope([0])),))


def test_regions_traversal_deduplicates_repeated_nodes():
    leaf0 = Region(Scope([0]))
    leaf1 = Region(Scope([1]))
    mid = Region(Scope([0, 1]))
    root = Region(Scope([0, 1]))

    mid.add_partition((leaf0, leaf1))
    # Unary partitions are valid and model region refinements in the DAG.
    root.add_partition((mid,))
    root.add_partition((leaf0, leaf1))

    # A DAG can reference the same region multiple ways; traversal returns unique nodes.
    rg = RegionGraph(root)
    seen = rg.regions
    assert seen == {root, mid, leaf0, leaf1}
