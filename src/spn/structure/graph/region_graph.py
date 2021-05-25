"""
Created on May 05, 2021

@authors: Kevin Huy Nguyen, Bennet Wittelsbach

This file provides the structure and construction algorithm for abstract RegionGraphs, which are 
used to build RAT-SPNs.
"""
import random
from typing import Any, Optional, Set, List


class RegionGraph:
    """The RegionGraph holds all regions and partitions over a set of random variables X.

    Attributes:
        root_region:
            A Region over the whole set of random variables, X, that holds R Partitions.
            (R: the number of replicas)
    """

    def __init__(self) -> None:
        self.root_region: Region
        # TODO: je nach Implementierung von split() + region_graph_to_spn() werden regions/partitions
        #       im RegionGraph benoetigt oder nicht. Fuer weitere Infos siehe Kommentar bei split()
        # self.regions: Set[Region] = set()
        # self.partitions: Set[Partition] = set()

    def __str__(self) -> str:
        return "RegionGraph over RV " + str(self.root_region.random_variables)

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
            A set of Partitions that are the children of the Region. Usually, each Region
            but the root_region has exactly 1 Partition as child.
    """

    def __init__(self, random_variables: Optional[Set[int]]) -> None:
        self.random_variables = random_variables if random_variables else set()
        self.partitions: Set[Partition] = set()

    def __str__(self) -> str:
        return "Region with RV: " + str(self.random_variables)

    def __repr__(self) -> str:
        return self.__str__()


class Partition:
    """A Partition is a set of Regions, where the scopes of all Regions are pairwise distinct.

    Attributes:
        regions:
            A set of Regions. In the standard implementation (see "Random Sum-Product Networks"
            by Peharz et. al), each Partition has exactly two Regions as children.
    """

    def __init__(self, regions: Optional[Set[Region]]) -> None:
        self.regions = regions if regions else set()

    def __str__(self) -> str:
        return "Partition over " + str(
            [region.random_variables for region in self.regions]
        )

    def __repr__(self) -> str:
        return self.__str__()


def random_region_graph(X: Set[int], depth: int, replicas: int) -> RegionGraph:
    """Creates a RegionGraph from a set of random variables X.

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

    Returns:
        A RegionGraph with a binary tree structure, consisting of alternating Regions and Partitions.

    Raises:
        ValueError: If any argument is invalid.
    """
    if X is None or len(X) < 2:
        raise ValueError("Need at least two random variables to build RegionGraph")
    if depth < 0:
        raise ValueError("Depth must not be negative")
    if replicas < 1:
        raise ValueError("Number of replicas must be at least 1")

    region_graph = RegionGraph()
    root_region = Region(random_variables=X)
    region_graph.root_region = root_region

    for r in range(0, replicas):
        split(region_graph, root_region, depth)

    return region_graph


def split(region_graph: RegionGraph, parent_region: Region, depth: int) -> None:
    """Splits a Region into balanced Partitions.

    Recursively builds up a binary tree structure of the RegionGraph. First, it splits the
    random variables of the parent_region, Y, into a Partition consisting of two balanced,
    distinct subsets of Y, and adds it as a child of the parent_region. Then, split() will
    be called onto each of the two subsets of Y until the maximum depth of the RegionGraph
    is reached, OR each Region consists of only 1 random variable,

    Args:
        region_graph:
            The RegionGraph which is built. (TODO: Omit or add Regions + Partitions to it.)
        parent_region:
            The Region whose random variables are to be partitioned. The parent of the created Partition.
        depth:
            The maximum depth of the RegionGraph until which split() will be recursively called.
    """
    # TODO: im Paper werden alle Partitionen und Regionen dem RegionGraph hinzugefuegt.
    #       Durch dessen Graphstruktur sind diese als Nachkommen bereits enthalten.
    #       Je nachdem, wie region_graph_to_spn() implementiert wird, muessen entweder Part./Reg.
    #       auch dem RegionGraph hinzugefuegt werden, oder RegionGraph kann als Parameter aus
    #       split() entfernt werden!
    split_index = len(parent_region.random_variables) // 2
    shuffle_random_variables = list(parent_region.random_variables)
    random.shuffle(shuffle_random_variables)
    region1 = Region(set(shuffle_random_variables[:split_index]))
    region2 = Region(set(shuffle_random_variables[split_index:]))

    partition = Partition({region1, region2})
    parent_region.partitions.add(partition)

    if depth > 1:
        if len(region1.random_variables) > 1:
            split(region_graph, region1, depth - 1)
        if len(region2.random_variables) > 1:
            split(region_graph, region2, depth - 1)


def _print_region_graph(region_graph: RegionGraph) -> None:
    """Prints a RegionGraph in DFS fashion.

    TODO: Needs to be moved to a test-module.
    """
    nodes: List[Any] = list(region_graph.root_region.partitions)
    print("RegionGraph: ", region_graph.root_region.random_variables)

    while nodes:
        node: Any = nodes.pop(0)

        if type(node) is Partition:
            scope: List[int] = [region.random_variables for region in node.regions]
            print("Partition: ", scope)
            nodes.extend(node.regions)

        elif type(node) is Region:
            print("Region: ", node.random_variables)
            nodes.extend(node.partitions)

        else:
            raise ValueError("Node must be Region or Partition")


if __name__ == "__main__":
    region_graph = random_region_graph(X=set(range(1, 8)), depth=2, replicas=2)
    _print_region_graph(region_graph)
