"""
Created on July 1, 2021

@authors: Philipp Deibert

This file provides the base backend version of RAT-SPNs.
"""
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.base.structure.rat.region_graph import RegionGraph, Partition, Region, random_region_graph
from spflow.base.structure.nodes.node import SPNSumNode
from spflow.base.structure.layers.layer import SPNSumLayer, SPNPartitionLayer, SPNHadamardLayer
from spflow.base.structure.layers.leaves.parametric.gaussian import GaussianLayer
from spflow.base.structure.module import Module

from typing import Union, Iterable, Optional


class RatSPN(Module):
    """Torch backend module for RAT-SPNs.

    Args:
        region_graph:
            Region graph representing the high-level structure of the RAT-SPN.
        n_root_nodes:
            Number of nodes in the root region.
        n_region_nodes:
            Number of sum ndoes per internal region.
        n_leaf_nodes:
            Number of leaf nodes per leaf region.
    """
    def __init__(
        self,
        region_graph: RegionGraph,
        n_root_nodes: int,
        n_region_nodes: int,
        n_leaf_nodes: int,
    ) -> None:
        super(RatSPN, self).__init__()

        self.n_root_nodes = n_root_nodes
        self.n_region_nodes = n_region_nodes
        self.n_leaf_nodes = n_leaf_nodes

        if n_root_nodes < 1:
            raise ValueError(f"Specified value of 'n_root_nodes' must be at least 1, but is {n_root_nodes}.")
        if n_region_nodes < 1:
            raise ValueError(f"Specified value for 'n_region_nodes' must be at least 1, but is {n_region_nodes}.")
        if n_leaf_nodes < 1:
            raise ValueError(f"Specified value for 'n_leaf_nodes' must be at least 1, but is {n_leaf_nodes}.")

        # create RAT-SPN from region graph
        self.from_region_graph(region_graph)

    def from_region_graph(self, region_graph: RegionGraph) -> Union[SPNSumLayer, GaussianLayer, SPNHadamardLayer]:
        """TODO"""

        def convert_partition(partition: Partition) -> SPNPartitionLayer:

            return SPNPartitionLayer(child_partitions=[
                [convert_region(region, n_nodes=self.n_region_nodes)] for region in partition.regions
            ])
        
        def convert_region(region: Region, n_nodes: int) -> Union[SPNSumLayer, SPNHadamardLayer, GaussianLayer]:

            # non-leaf region
            if region.partitions:
                return SPNSumLayer(children=[convert_partition(partition) for partition in region.partitions], n_nodes=n_nodes)
            # leaf region
            else:
                # split leaf scope into univariate ones and combine them element-wise
                if len(region.scope.query) > 1:
                    return SPNHadamardLayer(child_partitions=
                        [[GaussianLayer(Scope([rv], region.scope.evidence), n_nodes=self.n_leaf_nodes)] for rv in region.scope.query]
                    )
                # create univariate leaf region
                elif len(region.scope.query) == 1:
                    return GaussianLayer(region.scope, n_nodes=self.n_leaf_nodes)
                else:
                    raise ValueError("Query scope for region is empty and cannot be converted into appropriate RAT-SPN layer representation.")
        
        self.root_region = convert_region(region_graph.root_region, n_nodes=self.n_root_nodes)
        self.root_node = SPNSumNode(children=[self.root_region])

    @property
    def n_out(self):
        # return number of outputs
        return 1
    
    @property
    def scopes_out(self):
        return self.root_node.scopes_out


@dispatch(memoize=True)
def marginalize(layer: RatSPN, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[SPNSumLayer, Module, None]:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    raise NotImplementedError()