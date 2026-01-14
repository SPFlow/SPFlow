from __future__ import annotations

import dataclasses
from typing import List, Set, Tuple

from spflow.meta.data.scope import Scope


@dataclasses.dataclass(unsafe_hash=True)
class Region:
    """A Region in the Region Graph, representing a set of variables.

    Attributes:
        scope (Scope): The set of variables in this region.
        children (List[Tuple[Region, ...]]): List of partitions. Each partition
            is a tuple of disjoint child Regions that form this Region.
    """
    scope: Scope
    children: List[Tuple[Region, ...]] = dataclasses.field(default_factory=list, hash=False)

    def add_partition(self, regions: Tuple[Region, ...]) -> None:
        """Add a partition (decomposition) of this region."""
        # Region-graph partitions must be a disjoint decomposition of this region's scope.
        if not Scope.all_pairwise_disjoint([r.scope for r in regions]):
            raise ValueError(
                f"Partition regions must have pairwise disjoint scopes, got {[r.scope for r in regions]}."
            )

        child_scope = Scope.join_all([r.scope for r in regions])
        if not child_scope.equal_query(self.scope):
            raise ValueError(f"Partition scope {child_scope} does not match region scope {self.scope}.")

        self.children.append(regions)


class RegionGraph:
    """Region Graph structure representing hierarchical variable decomposition.
    
    Attributes:
        root (Region): The root region containing all variables.
    """

    def __init__(self, root: Region) -> None:
        self.root = root

    @property
    def regions(self) -> Set[Region]:
        """Return all regions in the graph via traversal."""
        visited = set()
        stack = [self.root]
        while stack:
            r = stack.pop()
            if r in visited:
                continue
            visited.add(r)
            for partition in r.children:
                for child in partition:
                    stack.append(child)
        return visited

    def post_order(self) -> List[Region]:
        """Return regions in post-order (leaves first)."""
        visited = set()
        result = []

        def _dfs(region: Region):
            if region in visited:
                return
            visited.add(region)
            for partition in region.children:
                for child in partition:
                    _dfs(child)
            result.append(region)

        _dfs(self.root)
        return result
