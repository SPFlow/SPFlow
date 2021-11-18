"""
Created on July 1, 2021

@authors: Philipp Deibert

This file provides the PyTorch version of RAT SPNs.
"""

from spflow.base.structure.nodes.node import ILeafNode, ISumNode, INode
from spflow.base.structure.rat import RegionGraph, Partition, Region
from spflow.base.structure.rat import RatSpn, construct_spn
from spflow.torch.structure.nodes.node import TorchLeafNode
from spflow.torch.structure.nodes.leaves.parametric import TorchGaussian
from spflow.torch.structure.module import TorchModule

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
from typing import List, Union, Dict

from multipledispatch import dispatch  # type: ignore
from spflow.base.learning.context import Context  # type: ignore
from spflow.base.structure.nodes.leaves.parametric import (
    Gaussian,
    get_scipy_object,
    get_scipy_object_parameters,
)


class _PartitionLayer(nn.Module):
    """torch.nn.Module representing a RAT-SPN partition layer

    Computes the products of all unique element-wise combinations across the child regions' outputs.

    Args:
        regions (List[Union[_RegionLayer, _LeafLayer]]): List of (internal or leaf) child regions.
    """

    def __init__(self, regions=List[Union["_RegionLayer", "_LeafLayer"]]) -> None:
        super(_PartitionLayer, self).__init__()

        self.regions = regions

        # register regions as child modules
        for i, region in enumerate(regions, start=1):
            self.add_module(f"region_{i}", region)

        # compute number of outputs
        self.num_out = np.prod([len(region) for region in regions])

    def __len__(self) -> int:
        # return number of outputs
        return self.num_out

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # get inputs recursively
        inputs = [region(data) for region in self.regions]

        batch_size = data.shape[0]

        out_batch = []
        input_batch = [[inp[i] for inp in inputs] for i in range(batch_size)]

        # multiply cartesian products (sum in log space) for each entry in batch
        for inputs_batch in input_batch:
            out_batch.append(torch.cartesian_prod(*inputs_batch).sum(dim=1))

        out = torch.vstack(out_batch)  # type: ignore

        return out


class _RegionLayer(nn.Module):
    """torch.nn.Module representing a RAT-SPN region layer

    Outputs are computed as a convex sum of the inputs where each output has its own weights.

    Args:
        num_nodes_region (int): number of sum nodes in the region.
        partitions (List[_PartitionLayer]): List of child partitions.
    """

    def __init__(self, num_nodes_region: int, partitions=List["_PartitionLayer"]) -> None:
        super(_RegionLayer, self).__init__()

        self.partitions = partitions

        # register partitions as child modules
        for i, partition in enumerate(partitions, start=1):
            self.add_module(f"partition_{i}", partition)

        # compute number of inputs and outputs
        self.num_out = num_nodes_region
        self.num_in = sum(len(partition) for partition in partitions)

        # create and register weight parameters for sum nodes
        self.register_parameter(
            "weight",
            Parameter(torch.full(size=(self.num_out, self.num_in), fill_value=1.0 / self.num_in)),
        )

    def __len__(self) -> int:
        # return number of outputs
        return self.num_out

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # get inputs recursively
        inputs = torch.hstack([partition(data) for partition in self.partitions])  # type: ignore

        # broadcast inputs per output node and weight them in log-space
        weighted_inputs = inputs.unsqueeze(1) + self.weight.log()  # type: ignore

        return torch.logsumexp(weighted_inputs, dim=-1)


class _LeafLayer(nn.Module):
    """torch.nn.Module representing a RAT-SPN leaf layer

    Contains a set of individual leaf nodes.

    Args:
        scope (List[int]): list of integers representing the variable scope.
        num_nodes_leaf (int): number of leaf nodes in the leaf layer.
    """

    def __init__(self, scope: List[int], num_nodes_leaf: int) -> None:
        super(_LeafLayer, self).__init__()

        self.scope = scope

        # compute number of outputs
        self.num_out = num_nodes_leaf

        # TODO: set correct distributions
        # generate leaf nodes
        self.leaf_nodes: List[TorchLeafNode] = [
            TorchGaussian(scope, 0.0, 1.0) for i in range(num_nodes_leaf)
        ]

        # register leaf nodes as child modules
        for i, leaf in enumerate(self.leaf_nodes, start=1):
            self.add_module(f"leaf_{i}", leaf)

    def __len__(self) -> int:
        # return number of outputs
        return self.num_out

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        inputs = [leaf(data) for leaf in self.leaf_nodes]
        # generate outputs from all leaf nodes
        outputs = torch.hstack(inputs)  # type: ignore

        return outputs


class TorchRatSpn(TorchModule):
    """Torch backend module for RAT-SPNs.

    Args:
        region_graph (RegionGraph): region graph representing the high-level structure of the RAT-SPN.
        num_nodes_root (int): number of nodes in the root region.
        num_nodes_region (int): number of sum ndoes per internal region.
        num_nodes_leaf (int): number of leaf nodes per leaf region.
    """

    def __init__(
        self,
        region_graph: RegionGraph,
        num_nodes_root: int,
        num_nodes_region: int,
        num_nodes_leaf: int,
    ) -> None:

        super(TorchRatSpn, self).__init__()

        self.region_graph = region_graph
        self.num_nodes_root = num_nodes_root
        self.num_nodes_region = num_nodes_region
        self.num_nodes_leaf = num_nodes_leaf

        if num_nodes_root < 1:
            raise ValueError("num_nodes_root must be at least 1")
        if num_nodes_region < 1:
            raise ValueError("num_nodes_region must be at least 1")
        if num_nodes_leaf < 1:
            raise ValueError("num_nodes_leaf must be at least 1")

        self.rg_layers: Dict[
            Union[Region, Partition], Union[_RegionLayer, _LeafLayer, _PartitionLayer]
        ] = {}

        # recursively compute RatSpn layers beginning with root region
        self.root_region = _RegionLayer(
            num_nodes_root,
            partitions=[
                self.partition(partition) for partition in region_graph.root_region.partitions
            ],
        )

        # store root region mapping in dictionary
        self.rg_layers[region_graph.root_region] = self.root_region

        # create weights for root node
        self.register_parameter(
            "root_node_weight",
            Parameter(
                torch.full(size=(1, num_nodes_root), fill_value=1.0 / num_nodes_root)
            ),  # , dtype=torch.float64))
        )

    def partition(self, partition: Partition) -> _PartitionLayer:
        """Returns a _PartitionLayer object from a region graph Partition.

        Args:
            partition (Partition): Partition.
        """
        region_layers: List[Union[_LeafLayer, _RegionLayer]] = []

        # iterate over child regions
        for region in partition.regions:

            # leaf region (has no child partition)
            if not region.partitions:
                # create leaf layer with corresponding scope and number of leaf nodes
                leaf_layer = _LeafLayer(list(region.random_variables), self.num_nodes_leaf)  # type: ignore

                region_layers.append(leaf_layer)

                # store region mapping in dictionary
                self.rg_layers[region] = leaf_layer

            # internal region
            else:
                # create region layer and recursively convert child partitions
                region_layer = _RegionLayer(
                    self.num_nodes_region,
                    partitions=[self.partition(p) for p in region.partitions],
                )

                region_layers.append(region_layer)

                # store region mapping in dictionary
                self.rg_layers[region] = region_layer

        # create partition layer with computed child region layers
        partition_layer = _PartitionLayer(regions=region_layers)

        # store partition mapping in dictionary
        self.rg_layers[partition] = partition_layer

        return partition_layer

    def __len__(self):
        # return number of outputs
        return 1

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        # get inputs recursively
        inputs = self.root_region(data)

        # broadcast inputs per output node and weight them
        weighted_inputs = inputs.unsqueeze(1) + self.root_node_weight.log()  # type: ignore

        return torch.logsumexp(weighted_inputs, dim=-1)


@dispatch(_RegionLayer, list)  # type: ignore[no-redef]
def _copy_region_parameters(src: _RegionLayer, dst: List[ISumNode]) -> None:
    """Copy parameters from region layer to region sum nodes.

    args:
        src (_RegionLayer): region layer to copy parameters from.
        dst (List[ISumNode]): list of region nodes to copy parameters to.
    """

    # number of sum nodes should match number of region layer outputs
    if not len(dst) == len(src):
        raise ValueError("Number of ISumNodes and number of outputs for _RegionLayer do not match.")

    # iterate over nodes and region weights
    for i, node in enumerate(dst):

        # number of children of sum nodes should match number of region layer inputs
        if not (len(node.weights) == len(node.children) == src.weight.data.shape[1]):
            raise ValueError(
                "Number of ISumNode children or weights and number of inputs for _RegionLayer do not match."
            )

        # assign region weight slice to node weights
        node.weights = src.weight.data[i, :].detach().cpu().tolist()


@dispatch(_LeafLayer, list)  # type: ignore[no-redef]
def _copy_region_parameters(src: _LeafLayer, dst: List[ILeafNode]) -> None:
    """Copy parameters from leaf layer to leaf nodes.

    args:
        src (_LeafLayer): leaf layer to copy parameters from.
        dst (List[ILeafNode]): list of leaf nodes to copy parameters to.
    """

    # number of sum nodes should match number of region layer outputs
    if not len(dst) == len(src):
        raise ValueError("Number of ILeafNodes and number of outputs for _LeafLayer do not match.")

    # iterate over nodes and region weights
    for node, torch_node in zip(dst, src.leaf_nodes):

        # TODO: check whether node and torch node types match (e.g. call toNodes and check types)

        # copy leaf parameters from torch node
        node.set_params(*torch_node.get_params())


@dispatch(list, _RegionLayer)  # type: ignore[no-redef]
def _copy_region_parameters(src: List[ISumNode], dst: _RegionLayer) -> None:
    """Copy parameters from region sum nodes to region layer.

    args:
        src (List[ISumNode]): list of sum nodes to copy parameters from.
        dst (_RegionLayer): region layer to copy parameters to.
    """

    # number of sum nodes should match number of region layer outputs
    if not len(dst) == len(src):
        raise ValueError("Number of ISumNodes and number of outputs for _RegionLayer do not match.")

    # iterate over nodes and region weights
    for i, node in enumerate(src):

        # number of children of sum nodes should match number of region layer inputs
        if not (len(node.weights) == len(node.children) == dst.weight.data.shape[1]):
            raise ValueError(
                "Number of ISumNode children or weights and number of inputs for _RegionLayer do not match."
            )

        # assign node weights to region weight slice
        dst.weight.data[i, :] = torch.tensor(node.weights)


@dispatch(list, _LeafLayer)  # type: ignore[no-redef]
def _copy_region_parameters(src: List[ILeafNode], dst: _LeafLayer) -> None:
    """Copy parameters from leaf nodes to leaf layer.

    args:
        src (List[ILeafNode]): list of leaf nodes to copy parameters from.
        dst (_LeafLayer): leaf layer to copy parameters to.
    """

    # there must be some sum nodes
    if len(dst) == 0:
        raise ValueError("No ISumNodes specified.")

    # number of sum nodes should match number of region layer outputs
    if not len(dst) == len(src):
        raise ValueError("Number of ILeafNodes and number of outputs for _LeafLayer do not match.")

    # iterate over nodes and region weights
    for node, torch_node in zip(src, dst.leaf_nodes):

        # TODO: check whether node and torch node types match (e.g. call toTorch and check types)

        # copy leaf parameters from node
        torch_node.set_params(*node.get_params())


@dispatch(TorchRatSpn)  # type: ignore[no-redef]
def toNodes(torch_rat: TorchRatSpn) -> RatSpn:
    # create RAT-SPN module using region graph (includes scopes)
    context = Context(
        parametric_types=[Gaussian] * len(torch_rat.region_graph.root_region.random_variables)
    )
    rat = RatSpn(
        torch_rat.region_graph,
        torch_rat.num_nodes_root,
        torch_rat.num_nodes_region,
        torch_rat.num_nodes_leaf,
        context,
    )

    # get all regions
    for region in torch_rat.region_graph.regions:

        # get region layer from torch RAT SPN
        region_layer: Union[_RegionLayer, _LeafLayer, _PartitionLayer] = torch_rat.rg_layers[region]

        # get region nodes from node RAT SPN
        region_nodes: List[INode] = rat.rg_nodes[region]

        # internal region
        if isinstance(region_layer, _RegionLayer):
            # all nodes need to be ISumNodes
            if not all([isinstance(node, ISumNode) for node in region_nodes]):
                raise ValueError("Internal region nodes must all be ISumNodes.")
        # leaf region
        else:
            # all nodes need to be ILeafNodes
            if not all([isinstance(node, ILeafNode) for node in region_nodes]):
                raise ValueError("Leaf region nodes must all be ILeafNodes.")

        # transfer region parameters from nodes to layer
        _copy_region_parameters(region_layer, region_nodes)

    # transfer root node weight
    rat.output_nodes[0].weights = torch_rat.root_node_weight.data.cpu().numpy()  # type: ignore

    return rat


@dispatch(RatSpn)  # type: ignore[no-redef]
def toTorch(rat: RatSpn) -> TorchRatSpn:
    torch_rat = TorchRatSpn(
        rat.region_graph, rat.num_nodes_root, rat.num_nodes_region, rat.num_nodes_leaf
    )

    # get all regions
    for region in rat.region_graph.regions:

        # get region layer from torch RAT SPN
        region_layer: Union[_RegionLayer, _LeafLayer, _PartitionLayer] = torch_rat.rg_layers[region]

        # get region nodes from node RAT SPN
        region_nodes: List[INode] = rat.rg_nodes[region]

        # internal region
        if isinstance(region_layer, _RegionLayer):
            # all nodes need to be ISumNodes
            if not all([isinstance(node, ISumNode) for node in region_nodes]):
                raise ValueError("Internal region nodes must all be ISumNodes.")
        # leaf region
        else:
            # all nodes need to be ILeafNodes
            if not all([isinstance(node, ILeafNode) for node in region_nodes]):
                raise ValueError("Leaf region nodes must all be ILeafNodes.")

        # transfer region parameters from layer to nodes
        _copy_region_parameters(region_nodes, region_layer)

    # transfer root node weight
    torch_rat.root_node_weight.data = torch.tensor(rat.output_nodes[0].weights)

    return torch_rat
