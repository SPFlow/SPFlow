# -*- coding: utf-8 -*-
"""Contains the SPFlow architecture for Random and Tensorized Sum-Product Networks (RAT-SPNs) in the ``torch`` backend.
"""
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.base.structure.rat.region_graph import RegionGraph, Partition, Region
from spflow.base.structure.rat.rat_spn import RatSPN as BaseRatSPN
from spflow.torch.structure.nodes.node import SPNSumNode
from spflow.torch.structure.layers.layer import SPNSumLayer, SPNPartitionLayer, SPNHadamardLayer
from spflow.torch.structure.layers.leaves.parametric.gaussian import GaussianLayer
from spflow.torch.structure.module import Module

from typing import Union, Iterable, Optional, List


class RatSPN(Module):
    r"""Module architecture for Random and Tensorized Sum-Product Networks (RAT-SPNs) in the ``torch`` backend.

    Constructs a RAT-SPN from a specified ``RegionGraph`` instance.
    For details see (Peharz et al., 2020): "Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning".

    Attributes:
        n_root_nodes:
            Integer specifying the number of sum nodes in the root region (C in the original paper).
        n_region_nodes:
            Integer specifying the number of sum nodes in each (non-root) region (S in the original paper).
        n_leaf_ndoes:
            Integer specifying the number of leaf nodes in each leaf region (I in the original paper).
        root_node:
            SPN-like sum node that represents the root of the model.
        root_region:
            SPN-like sum layer that represents the root region of the model.
    """
    def __init__(
        self,
        region_graph: RegionGraph,
        n_root_nodes: int,
        n_region_nodes: int,
        n_leaf_nodes: int,
    ) -> None:
        super(RatSPN, self).__init__(children=[])
        r"""Initializer for ``RatSPN`` object.

        Args:
            region_graph:
                ``RegionGraph`` instance to create RAT-SPN architecture from.
            n_root_nodes:
                Integer specifying the number of sum nodes in the root region (C in the original paper).
            n_region_nodes:
                Integer specifying the number of sum nodes in each (non-root) region (S in the original paper).
            n_leaf_ndoes:
                Integer specifying the number of leaf nodes in each leaf region (I in the original paper).

        Raises:
            ValueError: Invalid arguments.
        """
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
        r"""Function to create explicit RAT-SPN from an abstract region graph.

        Args:
            region_graph:
                ``RegionGraph`` instance to create RAT-SPN architecture from.
        Returns:
            ValueError: Invalid arguments.
        """
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

        if region_graph.root_region is not None:
            self.root_region = convert_region(region_graph.root_region, n_nodes=self.n_root_nodes)
            self.root_node = SPNSumNode(children=[self.root_region])
        else:
            self.root_region = None
            self.root_node = None
        
    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Returns one since RAT-SPNs always have a single output."""
        return 1
    
    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes of the RAT-SPN."""
        return self.root_node.scopes_out


@dispatch(memoize=True)   # type: ignore
def marginalize(rat_spn: RatSPN, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[RatSPN, Module, None]:
    r"""Structural marginalization for ``RatSPN`` objects in the ``torch`` backend.

    Raises a ``NoteImplementedError`` since structural marginalization is not yet supported for RAT-SPNs.

    TODO

    Args:
        rat_spn:
           ``RatSPN`` instance to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        NotImplementedError: Structural marginalization is not yet supported for RAT-SPNs.    
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    raise NotImplementedError()


@dispatch(memoize=True)  # type: ignore
def toBase(rat_spn: RatSPN, dispatch_ctx: Optional[DispatchContext]=None) -> BaseRatSPN:
    r"""Conversion for ``RatSPN`` from ``torch`` backend to ``base`` backend.
    
    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Optional dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # create RAT-SPN in base backend (using empty region graph)
    base_rat_spn = BaseRatSPN(RegionGraph(), n_root_nodes=rat_spn.n_root_nodes, n_region_nodes=rat_spn.n_region_nodes, n_leaf_nodes=rat_spn.n_leaf_nodes)

    # replace actual module graph
    base_rat_spn.root_node = toBase(rat_spn.root_node, dispatch_ctx=dispatch_ctx)
    # set root region
    base_rat_spn.root_region = base_rat_spn.root_node.children[0]

    return base_rat_spn


@dispatch(memoize=True)  # type: ignore
def toTorch(rat_spn: BaseRatSPN, dispatch_ctx: Optional[DispatchContext]=None) -> RatSPN:
    r"""Conversion for ``RatSPN`` from ``base`` backend to ``torch`` backend.
    
    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Optional dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # create RAT-SPN in base backend (using empty region graph)
    torch_rat_spn = RatSPN(RegionGraph(), n_root_nodes=rat_spn.n_root_nodes, n_region_nodes=rat_spn.n_region_nodes, n_leaf_nodes=rat_spn.n_leaf_nodes)

    # replace actual module graph
    torch_rat_spn.root_node = toTorch(rat_spn.root_node, dispatch_ctx=dispatch_ctx)
    # set root region
    torch_rat_spn.root_region = next(torch_rat_spn.root_node.children())

    return torch_rat_spn