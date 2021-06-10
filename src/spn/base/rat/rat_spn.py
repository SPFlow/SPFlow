"""
Created on May 24, 2021

@authors: Bennet Wittelsbach
"""
import itertools
from functools import reduce
from typing import List, cast

from spn.base.module import Module
from spn.base.nodes.node import LeafNode, ProductNode, SumNode, _print_node_graph
from spn.base.rat.region_graph import (
    RegionGraph,
    _print_region_graph,
    random_region_graph,
)


class RatSpn(Module):
    """A RAT-SPN is a randomized SPN, usually built from a RegionGraph.

    Attributes:
        root_nodes:
            A list of SumNodes that are the roots of the RAT-SPN. Root nodes are the outputs
            of SPNs. Usually, SPNs only have one root node, but can also have multiple roots
            for multiple outputs, e.g. classes. When the SPN is constructed from a
            RegionGraph, the roots are the nodes of the root_region of the RegionGraph.

    """

    def __init__(self) -> None:
        self.root_nodes: List[SumNode] = list()
    
    def __len__(self):
        return 1


def construct_spn(
    region_graph: RegionGraph,
    num_nodes_root: int,
    num_nodes_region: int,
    num_nodes_leaf: int,
) -> RatSpn:
    """Builds a RAT-SPN from a given RegionGraph.

    This algorithm is an implementation of "Algorithm 2" of the original paper. The Regions and
    Partitions in the RegionGraph are equipped with an appropriate number of nodes each, and the
    nodes will be connected afterwards. The resulting RAT-SPN holds a list of the root nodes, which
    in turn hold the whole constructed (graph) SPN. The number of ProductNodes in a Partition is
    determined by the length of the cross product of the children Regions of the respective Partition.

    Args:
        num_nodes_root:
            (C in the paper)
            The number of SumNodes the root_region is equipped with. This will be the length of the
            root_node list of the resulting RAT-SPN and the number of output nodes, respectively.
        num_nodes_region:
            (S in the paper)
            The number of SumNodes each region but the root and leaf regions are equipped with.
        num_nodes_leaf:
            (I in the paper)
            The number of LeafNodes each leaf region is equipped with. All LeafNodes of the same region
            are multivariate distributions over the same scope, but possibly differently parametrized.
    """
    rat_spn = RatSpn()

    for region in region_graph.regions:
        # determine the scope of the nodes the Region will be equipped with
        region_scope = list(region.random_variables)
        region_scope.sort()
        if not region.parent:
            # the region is the root_region
            region.nodes = [
                SumNode(children=[], scope=region_scope, weights=[])
                for i in range(num_nodes_root)
            ]
            rat_spn.root_nodes = cast(List[SumNode], region.nodes)
        elif not region.partitions:
            # the region is a leaf
            region.nodes = [LeafNode(scope=region_scope) for i in range(num_nodes_leaf)]
        else:
            # the region is an internal region
            region.nodes = [
                SumNode(children=[], scope=region_scope, weights=[])
                for i in range(num_nodes_region)
            ]

    for partition in region_graph.partitions:
        # determine the number and the scope of the ProductNodes the Partition will be equipped with
        num_nodes_partition = reduce(
            (lambda r, s: r * s), [len(region.nodes) for region in partition.regions]
        )
        partition_scope = list(
            itertools.chain(*[region.random_variables for region in partition.regions])
        )
        partition_scope.sort()
        partition.nodes = [
            ProductNode(children=[], scope=partition_scope)
            for i in range(num_nodes_partition)
        ]

        # each ProductNode of the Partition points to a unique combination consisting of one Node of each Region that is a child of the partition
        cartesian_product = list(
            itertools.product(*[region.nodes for region in partition.regions])
        )
        for i in range(len(cartesian_product)):
            partition.nodes[i].children = list(cartesian_product[i])
        # all ProductNodes of the Partition are children of each SumNode in its parent Region
        for parent_node in partition.parent.nodes:
            # all parent nodes are SumNodes
            parent_node = cast(SumNode, parent_node)
            replicas = len(partition.parent.partitions)
            parent_node.children.extend(partition.nodes)
            # determine the total number of children the parent node might have. this is important for correct weights in the root nodes
            parent_node.weights.extend(
                [1 / (num_nodes_partition * replicas)] * num_nodes_partition
            )

    if not rat_spn.root_nodes:
        raise ValueError("Constructed RAT-SPN does not have root nodes")

    return rat_spn


if __name__ == "__main__":
    region_graph = random_region_graph(X=set(range(1, 8)), depth=2, replicas=2)
    _print_region_graph(region_graph)
    rat_spn = construct_spn(region_graph, 1, 2, 2)
    _print_node_graph(rat_spn.root_nodes)
