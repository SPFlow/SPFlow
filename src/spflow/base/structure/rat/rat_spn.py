"""
Created on May 24, 2021

@authors: Bennet Wittelsbach
"""
import itertools
import numpy as np
from typing import Dict, List, Union, cast, Optional, Tuple, Type, Sequence
from spflow.base.structure.module import Module
from spflow.base.learning.context import Context  # type: ignore
from spflow.base.structure.nodes.node import INode, IProductNode, ISumNode, get_topological_order
from spflow.base.structure.network_type import SPN
from spflow.base.structure.nodes.leaves.parametric import MultivariateGaussian
from .region_graph import (
    Partition,
    Region,
    RegionGraph,
)


class RatSpn(Module):
    """A RAT-SPN is a randomized SPN, usually built from a RegionGraph. This is the module implementation for RAT-SPNs.
       The paper referenced: Random Sum-Product Networks A Simple but Effective Approach to Probabilistic Deep Learning
                          by R. Peharz et. al.

    Attributes:
        nodes:
            List of nodes in RAT-SPN sorted in topological order bottom up.
        output_nodes:
            Output INodes of RAT-SPN. In case of the RAT-SPN only one node is needed to describe the entire RAT-SPN.
            In the following that node is referred to as root node.
        rg_nodes:
            Dictionary with k: regions and v: corresponding nodes in the region.
        region_graph:
            Region graph, the RAT-SPN is created from.
        num_nodes_root:
            (C in the paper)
            The number of ISumNodes the root_region is equipped with. This will be the length of the children of the
            root_node of the resulting RAT-SPN.
        num_nodes_region:
            (S in the paper)
            The number of ISumNodes each region except the root and leaf regions are equipped with.
        num_nodes_leaf:
            (I in the paper)
            The number of LeafNodes each leaf region is equipped with. All LeafNodes of the same region
            are multivariate distributions over the same scope, but possibly differently parametrized.
    """

    def __init__(
        self,
        region_graph: RegionGraph,
        num_nodes_root: int,
        num_nodes_region: int,
        num_nodes_leaf: int,
        context: Context,
    ) -> None:
        rat_result: Tuple[ISumNode, Dict[Union[Region, Partition], List[INode]]] = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )
        super().__init__(children=[], network_type=SPN(), scope=rat_result[0].scope)
        rat_spn: ISumNode = rat_result[0]
        rat_spn_rg_nodes: Dict[Union[Region, Partition], List[INode]] = rat_result[1]
        self.nodes: List[INode] = get_topological_order(rat_spn)
        self.output_nodes: List[ISumNode] = [rat_spn]

        # RAT-module specific attributes
        self.region_graph: RegionGraph = region_graph
        self.num_nodes_root: int = num_nodes_root
        self.num_nodes_region: int = num_nodes_region
        self.num_nodes_leaf: int = num_nodes_leaf
        self.rg_nodes: Dict[Union[Region, Partition], List[INode]] = rat_spn_rg_nodes

    def __len__(self):
        return 1


def construct_spn(
    region_graph: RegionGraph,
    num_nodes_root: int,
    num_nodes_region: int,
    num_nodes_leaf: int,
    context: Context,
) -> Tuple[ISumNode, Dict[Union[Region, Partition], List[INode]]]:
    """Builds a RAT-SPN from a given RegionGraph.

    This algorithm is an implementation of "Algorithm 2" of the original paper. The Regions and
    Partitions in the RegionGraph are equipped with an appropriate number of nodes each, and the
    nodes will be connected afterwards. The resulting RAT-SPN holds a list of the root nodes, which
    in turn hold the whole constructed (graph) SPN. The number of IProductNodes in a Partition is
    determined by the length of the cross product of the children Regions of the respective Partition.

    Args:
        region_graph:
            Region graph, the RAT-SPN is created from.
        num_nodes_root:
            (C in the paper)
            The number of ISumNodes the root_region is equipped with. This will be the length of the children of the
            root_node of the resulting RAT-SPN.
        num_nodes_region:
            (S in the paper)
            The number of ISumNodes each region except the root and leaf regions are equipped with.
        num_nodes_leaf:
            (I in the paper)
            The number of LeafNodes each leaf region is equipped with. All LeafNodes of the same region
            are multivariate distributions over the same scope, but possibly differently parametrized.
        context:
            Context object determining distributions for scopes in SPN.


    Returns:
        A tuple with two entries:
        The first entry is a RatSpn with a single ISumNode as root. It's children are the ISumNodes of the root_region
        in the region_graph. The rest of the SPN consists of alternating Sum- and IProductNodes, providing
        the scope factorizations determined by the region_graph.
        The second entry is a dictionary with k: regions and v: corresponding nodes in the region.

    Raises:
        ValueError:
            If any argument is invalid (too few roots to build an SPN).
    """
    if num_nodes_root < 1:
        raise ValueError("num_nodes_root must be at least 1")
    if num_nodes_region < 1:
        raise ValueError("num_nodes_region must be at least 1")
    if num_nodes_leaf < 1:
        raise ValueError("num_nodes_leaf must be at least 1")

    rg_nodes: Dict[Union[Region, Partition], List[INode]] = {}
    root_node = None

    for region in region_graph.regions:
        # determine the scope of the nodes the Region will be equipped with
        region_scope = list(region.random_variables)
        region_scope.sort()
        if not region.parent:
            # the region is the root_region
            root_nodes: List[INode] = [
                ISumNode(children=[], scope=region_scope, weights=np.empty(0))
                for i in range(num_nodes_root)
            ]
            rg_nodes[region] = root_nodes
            root_node = ISumNode(
                children=root_nodes,
                scope=region_scope,
                weights=np.full(len(rg_nodes[region]), 1 / len(rg_nodes[region])),
            )
        elif not region.partitions:
            # the region is a leaf
            rg_nodes[region] = [
                context.parametric_types[region_scope[0]](scope=region_scope)
                if len(region_scope) == 1
                else MultivariateGaussian(
                    scope=region_scope,
                    mean_vector=np.zeros(len(region_scope)),
                    covariance_matrix=np.eye(len(region_scope)),
                )
                for i in range(num_nodes_leaf)
            ]
        else:
            # the region is an internal region
            rg_nodes[region] = [
                ISumNode(children=[], scope=region_scope, weights=np.empty(0))
                for i in range(num_nodes_region)
            ]

    for partition in region_graph.partitions:
        # determine the number and the scope of the IProductNodes the Partition will be equipped with
        num_nodes_partition = np.prod([len(rg_nodes[region]) for region in partition.regions])

        partition_scope = list(
            itertools.chain(*[region.random_variables for region in partition.regions])
        )
        partition_scope.sort()
        rg_nodes[partition] = [
            IProductNode(children=[], scope=partition_scope) for i in range(num_nodes_partition)
        ]

        # each IProductNode of the Partition points to a unique combination consisting of one INode of each Region
        # that is a child of the partition
        cartesian_product = list(
            itertools.product(*[rg_nodes[region] for region in partition.regions])
        )
        for i in range(len(cartesian_product)):
            rg_nodes[partition][i].children = list(cartesian_product[i])
        # all IProductNodes of the Partition are children of each ISumNode in its parent Region
        for parent_node in rg_nodes[partition.parent]:
            # all parent nodes are ISumNodes
            parent_node = cast(ISumNode, parent_node)
            replicas = len(partition.parent.partitions)
            parent_node.children.extend(rg_nodes[partition])
            # determine the total number of children the parent node might have.
            # this is important for correct weights in the root nodes
            parent_node.weights = np.append(
                parent_node.weights,
                np.full(num_nodes_partition, 1 / (num_nodes_partition * replicas)),
            )

    if not root_node:
        raise ValueError("Constructed RAT-SPN does not have root node")

    return root_node, rg_nodes
