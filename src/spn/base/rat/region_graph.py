"""
Created on May 05, 2021

@authors: Kevin Huy Nguyen, Bennet Wittelsbach

This file provides the structure and construction algorithm for abstract RegionGraphs, which are
used to build RAT-SPNs.
"""
import random
import numpy as np
from typing import Any, Optional, Set, List, Tuple

from spn.base.nodes.node import Node, ProductNode


class RegionGraph:
    """The RegionGraph holds all regions and partitions over a set of random variables X.

    Attributes:
        root_region:
            A Region over the whole set of random variables, X, that holds r Partitions.
            (r: the number of replicas)
        regions:
            A set of all Regions in the RegionGraph.
        partitions:
            A set of all Partitions in the RegionGraph.
    """

    def __init__(self, rnd_seed: int = 12345) -> None:
        self.root_region: Region
        self.regions: Set[Region] = set()
        self.partitions: Set[Partition] = set()
        random.seed(rnd_seed)

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
        children:
            A set of Partitions that are the children of the Region. Usually, each Region but the
            root_region has exactly 1 Partition as child. A leaf Region has no children at all.
        parent:
            The parent Partition. If the Region has no parent, it is the root region.
        nodes:
            A list of SumNodes or LeafNodes assigned while constructing the RATSPN.
    """

    def __init__(
        self,
        random_variables: Set[int],
        partitions: Optional[Set["Partition"]],
        parent: Optional["Partition"],
    ) -> None:
        self.random_variables = random_variables
        self.partitions = partitions if partitions else set()
        self.parent = parent
        self.nodes: List[Node] = []

    def __str__(self) -> str:
        return f"Region: {self.random_variables}"

    def __repr__(self) -> str:
        return self.__str__()


class Partition:
    """A Partition is a set of Regions, where the scopes of all Regions are pairwise distinct.

    Attributes:
        children:
            A set of Regions. In the standard implementation (see "Random Sum-Product Networks"
            by Peharz et. al), each Partition has exactly two Regions as children.
        parent:
            The parent Region. Each partition has exactly one parent Region.
        nodes:
            A list of ProductNodes assigned while constructing the RAT-SPN.
    """

    def __init__(self, regions: Set[Region], parent: Region) -> None:
        self.regions = regions
        self.parent = parent
        self.nodes: List[ProductNode] = []

    def __str__(self) -> str:
        return f"Partition: {[region.random_variables for region in self.regions]}"

    def __repr__(self) -> str:
        return self.__str__()


def random_region_graph(X: Set[int], depth: int, replicas: int, num_splits: int = 2) -> RegionGraph:
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
    if X is None or len(X) < num_splits:
        raise ValueError("Need at least 'num_splits' random variables to build RegionGraph")
    if depth < 0:
        raise ValueError("Depth must not be negative")
    if replicas < 1:
        raise ValueError("Number of replicas must be at least 1")
    if num_splits < 2:
        raise ValueError("Number of splits must be at least 2")

    region_graph = RegionGraph()
    root_region = Region(random_variables=X, partitions=None, parent=None)
    region_graph.root_region = root_region
    region_graph.regions.add(root_region)

    for r in range(0, replicas):
        split(region_graph, root_region, depth, num_splits)

    return region_graph


def split(
    region_graph: RegionGraph, parent_region: Region, depth: int, num_splits: int = 2
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

    shuffle_random_variables = list(parent_region.random_variables)
    random.shuffle(shuffle_random_variables)

    splits = np.array_split(shuffle_random_variables, num_splits)

    if any(region_scope.size == 0 for region_scope in splits):
        raise ValueError(
            "Number of random variables cannot be split into 'num_splits' "
            "non-empty splits (make sure 'split' is called appropriately)."
        )

    regions = []

    for region_scope in splits:
        regions.append(Region(random_variables=set(region_scope), partitions=None, parent=None))

    partition = Partition(regions={*regions}, parent=parent_region)

    for region in regions:
        region.parent = partition

    parent_region.partitions.add(partition)

    region_graph.partitions.add(partition)
    region_graph.regions.update(regions)

    if depth > 1:
        for region in regions:
            # if region scope can be divided into 'num_splits' non-empty splits
            if len(region.random_variables) >= num_splits:
                split(region_graph, region, depth - 1, num_splits)


def _print_region_graph(region_graph: RegionGraph) -> None:
    """Prints a RegionGraph in BFS fashion.

    Args:
        region_graph:
            A RegionGraph that holds a root_region
    """
    nodes: List[Any] = [region_graph.root_region]
    n_regions = 0
    n_partitions = 0

    while nodes:
        node: Any = nodes.pop(0)
        print(node)

        if type(node) is Region:
            n_regions += 1
            nodes.extend(node.partitions)
        elif type(node) is Partition:
            n_partitions += 1
            nodes.extend(node.regions)
        else:
            raise ValueError("Node must be Region or Partition")

    print(f"#Regions: {n_regions}, #Partitions: {n_partitions}")


def _get_regions_by_depth(
    region_graph: RegionGraph,
) -> Tuple[List[List[Region]], List[Region]]:
    """
    Returns:
        A 2-dimensional List of Regions, where the first index is the depth in the RegionGraph, and
        the second index points to the List of Regions of that depth; and a List of all leaf-Regions.
        example A: RegionGraph(X=[1, 2, 3], depth=2, replicas=1):
        ([
            [{1, 2, 3}],
            [{1, 2}, {3}],
            [{1}, {2}]
        ], [
            {1},
            {2},
            {3}
        ])

        example B: RegionGraph(X=[1, 2, 3, 4, 5, 6, 7], depth=2, replicas=1):
        ([
            [{1, 2, 3, 4, 5, 6, 7}],
            [{1, 3, 5, 7}, {2, 4, 6}],
            [{1, 7}, {3, 5}, {2}, {4, 6}]
        ], [
            {1, 7},
            {3, 5},
            {2},
            {4, 6}
        ])

        A
    """
    depth = 0
    regions_by_depth = [[region_graph.root_region]]
    leaves = []

    nodes: List[Any] = list(region_graph.root_region.partitions)

    while nodes:
        nodeCount = len(nodes)

        # increase depth by one for every Region-"layer" encountered
        peek = nodes[0]
        if type(peek) is Region:
            depth += 1
            regions_by_depth.append([])

        while nodeCount > 0:
            # process the whole "layer" of Regions/Partitions in the nodes-list
            node: Any = nodes.pop(0)
            nodeCount -= 1

            if type(node) is Partition:
                nodes.extend(node.regions)
            elif type(node) is Region:
                regions_by_depth[depth].append(node)
                if node.partitions:
                    nodes.extend(node.partitions)
                else:
                    # if the Region has no children, it is a leaf
                    leaves.append(node)
            else:
                raise ValueError("Node must be Region or Partition")

    return regions_by_depth, leaves


if __name__ == "__main__":
    region_graph = random_region_graph(X=set(range(1, 8)), depth=3, replicas=1)
    _print_region_graph(region_graph)
