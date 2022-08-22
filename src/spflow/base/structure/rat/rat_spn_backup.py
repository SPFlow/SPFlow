"""
Created on August 12, 2022

@authors: Philipp Deibert

This file provides the base version of RAT SPNs.
"""
import numpy as np
from typing import Dict, List, Union, cast, Tuple
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.node import Node, SPNProductNode, SPNSumNode
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from .region_graph import Partition, Region, RegionGraph


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
    ) -> None:
        # RAT-module specific attributes
        self.region_graph: RegionGraph = region_graph
        self.num_nodes_root: int = num_nodes_root
        self.num_nodes_region: int = num_nodes_region
        self.num_nodes_leaf: int = num_nodes_leaf

        rat_result: Tuple[SPNSumNode, Dict[Union[Region, Partition], List[Node]]] = self._construct_spn()
        
        super().__init__(children=[], scope=rat_result[0].scope)
        
        rat_spn: SPNSumNode = rat_result[0]
        rat_spn_rg_nodes: Dict[Union[Region, Partition], List[Node]] = rat_result[1]

        self.rg_nodes: Dict[Union[Region, Partition], List[Node]] = rat_spn_rg_nodes

    def _construct_spn(self) -> None:

        if self.num_nodes_root < 1:
            raise ValueError("num_nodes_root must be at least 1")
        if self.num_nodes_region < 1:
            raise ValueError("num_nodes_region must be at least 1")
        if self.num_nodes_leaf < 1:
            raise ValueError("num_nodes_leaf must be at least 1")

        rg_nodes: Dict[Union[Region, Partition], List[Node]] = {}
        root_node = None

        for region in self.region_graph.regions:
            # determine the scope of the nodes the Region will be equipped with
            region_scope = region.scope

            if not region.parent:
                # the region is the root_region
                root_nodes: List[Node] = [
                    SPNSumNode(children=[], weights=np.empty(0)) # TODO: wtf !?
                    for i in range(self.num_nodes_root)
                ]
                rg_nodes[region] = root_nodes
                root_node = SPNSumNode(
                    children=root_nodes,
                    weights=np.full(len(rg_nodes[region]), 1 / len(rg_nodes[region])),
                )
            elif not region.partitions:
                # the region is a leaf
                rg_nodes[region] = [
                    Gaussian(scope=region_scope)
                    if len(region_scope.query) == 1
                    else SPNProductNode(
                        children=[Gaussian(scope=Scope([rv], region.scope.evidence)) for rv in region_scope.query],
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

    @property
    def n_out(self) -> int:
        return 1
    
    @property
    def scopes_out(self) -> List[Scope]:
        pass