"""
Created on May 05, 2021

@authors: Kevin Huy Nguyen, Bennet Wittelsbach, Philipp Deibert

This file provides the structure and construction algorithm for abstract RegionGraphs, which are
used to build RAT-SPNs.
"""
import random
import numpy as np
from typing import Optional, Set, List
from spflow.meta.scope.scope import Scope


class RegionGraph:
    """Abstract region and partition graph used in the construction of RAT-SPNs.

    TODO

    Attributes:
        root_region:
            A Region over the whole set of random variables, X, that holds r Partitions.
            (r: the number of replicas)
        regions:
            A set of all Regions in the RegionGraph.
        partitions:
            A set of all Partitions in the RegionGraph.
    """

    def __init__(self) -> None:
        self.root_region: Region = None
        self.regions: List[Region] = list()
        self.partitions: List[Partition] = list()

    def __str__(self) -> str:
        return f"RegionGraph over RV {self.root_region.random_variables}"

    def __repr__(self) -> str:
        return self.__str__()


class Region:
    """A Region is a non-empty subset of the random variables X.

    All leafs of a RegionGraph are Regions with no children (empty set of Partitions).
    All Regions that are not leafs and not the root_region have exactly one Partition as child.

    Attributes:
        random_variables:
            A set of random variables in the scope of the Region.
        partitions:
            A list of Partitions that are the children of the Region. Usually, each Region but the
            root_region has exactly 1 Partition as child. A leaf Region has no children at all.
        parent:
            The parent Partition. If the Region has no parent, it is the root region.
    """

    def __init__(
        self,
        scope: Scope,
        partitions: Optional[List["Partition"]],
        parent: Optional["Partition"],
    ) -> None:
        self.scope = scope
        self.partitions = partitions if partitions else list()
        self.parent = parent

    def __str__(self) -> str:
        return f"Region: {self.scope}"

    def __repr__(self) -> str:
        return self.__str__()


class Partition:
    """A Partition is a set of Regions, where the scopes of all Regions are pairwise distinct.

    Attributes:
        regions:
            A set of Regions. In the standard implementation (see "Random Sum-Product Networks"
            by Peharz et. al), each Partition has exactly two Regions as children.
        parent:
            The parent Region. Each partition has exactly one parent Region.
    """

    def __init__(self, regions: List[Region], parent: Region) -> None:
        self.regions = regions
        self.parent = parent

    def __str__(self) -> str:
        return f"Partition: {[region.scope for region in self.regions]}"

    def __repr__(self) -> str:
        return self.__str__()


def random_region_graph(
    scope: Scope, depth: int, replicas: int, num_splits: int = 2
) -> RegionGraph:
    """Creates a RegionGraph from a set of random variables X.

    This algorithm is an implementation of "Algorithm 1" of the original paper.

    Args:
        X:
            The set of all indices/scopes of random variables that the RegionGraph contains.
        depth:
            (D in the paper)
            An integer that controls the depth of the graph structure of the RegionGraph. One level
            of depth equals to a pair of (Partitions, Regions). The root has depth 0.
        replicas:
            (R in the paper)
            An integer for the number of replicas. Replicas are distinct Partitions of the whole
            set of random variables X, which are children of the root_region of the RegionGraph.
        num_splits:
            The number of splits per Region (defaults to 2).

    Returns:
        A RegionGraph with a binary tree structure, consisting of alternating Regions and Partitions.

    Raises:
        ValueError: If any argument is invalid.
    """
    if (scope.query) < num_splits:
        raise ValueError("Need at least 'num_splits' query RVs to build region graph.")
    if depth < 0:
        raise ValueError("Depth must not be negative.")
    if replicas < 1:
        raise ValueError("Number of replicas must be at least 1.")
    if num_splits < 2:
        raise ValueError("Number of splits must be at least 2.")

    region_graph = RegionGraph()
    root_region = Region(scope=scope, partitions=None, parent=None)
    region_graph.root_region = root_region
    region_graph.regions.append(root_region)

    for r in range(0, replicas):
        split(region_graph, root_region, depth, num_splits)

    return region_graph


def split(
    region_graph: RegionGraph,
    parent_region: Region,
    depth: int,
    num_splits: int = 2
) -> None:
    """Splits a Region into (currently balanced) Partitions.

    Recursively builds up a binary tree structure of the RegionGraph. First, it splits the
    random variables of the parent_region, Y, into a Partition consisting of two balanced,
    distinct subsets of Y, and adds it as a child of the parent_region. Then, split() will
    be called onto each of the two subsets of Y until the maximum depth of the RegionGraph
    is reached, OR each Region consists of only 1 random variable,

    Args:
        region_graph:
            The RegionGraph which is built.
        parent_region:
            The Region whose random variables are to be partitioned. The parent of the created Partition.
        depth:
            The maximum depth of the RegionGraph until which split() will be recursively called.
        num_splits:
            The number of splits per Region.
    """
    if num_splits < 2:
        raise ValueError("Number of splits must be at least 2")

    shuffled_scope = parent_region.scope.query.copy()
    random.shuffle(shuffled_scope)

    splits = np.array_split(shuffled_scope, num_splits)

    if any(region_scope.size == 0 for region_scope in splits):
        raise ValueError(
            "Number of random variables cannot be split into 'num_splits' "
            "non-empty splits (make sure 'split' is called appropriately)."
        )

    regions = []

    for region_scope in splits:
        regions.append(Region(scope=Scope(region_scope), partitions=None, parent=None))

    partition = Partition(regions=regions, parent=parent_region)

    for region in regions:
        region.parent = partition

    parent_region.partitions.append(partition)

    region_graph.partitions.append(partition)
    region_graph.regions += regions

    if depth > 1:
        for region in regions:
            # if region scope can be divided into 'num_splits' non-empty splits
            if len(region.scope.query) >= num_splits:
                split(region_graph, region, depth - 1, num_splits)