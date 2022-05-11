"""
Created on July 1, 2021

@authors: Philipp Deibert

This file provides the PyTorch version of RAT SPNs.
"""

from spflow.base.structure.nodes.node import ILeafNode, ISumNode, INode
from spflow.base.structure.rat import RatSpn, RegionGraph, Partition, Region
from spflow.torch.structure.nodes.node import (
    TorchLeafNode,
    proj_real_to_convex,
    proj_convex_to_real,
)
from spflow.torch.structure.nodes.leaves.parametric import TorchGaussian, TorchMultivariateGaussian
from spflow.torch.structure.module import TorchModule

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
from typing import List, Union, Dict

from multipledispatch import dispatch  # type: ignore
from spflow.base.learning.context import RandomVariableContext  # type: ignore
from spflow.base.structure.nodes.leaves.parametric import Gaussian


class _TorchPartitionLayer(TorchModule):
    """TorchModule representing a RAT-SPN partition layer

    Computes the products of all unique element-wise combinations across the child regions' outputs.

    Args:
        regions:
            List of (internal or leaf) child region layers.
    """

    def __init__(
        self, children: List[Union["_TorchRegionLayer", "_TorchLeafLayer"]], scope: List[int]
    ) -> None:
        super(_TorchPartitionLayer, self).__init__()

        self.scope = scope

        # register regions as child modules
        for i, region in enumerate(children, start=1):
            self.add_module(f"region_{i}", region)

        # compute number of outputs
        self.num_out = np.prod([len(region) for region in self.children()])

    def __len__(self) -> int:
        # return number of outputs
        return self.num_out

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:

        batch_size = inputs[0].shape[0]

        out_batch = []
        input_batch = [[inp[i] for inp in inputs] for i in range(batch_size)]

        # multiply cartesian products (sum in log space) for each entry in batch
        for inputs_batch in input_batch:
            # try:
            out_batch.append(torch.cartesian_prod(*inputs_batch).sum(dim=1))
            # except:
            #    print(inputs_batch)
            #    input()
            #    raise Exception()

        out = torch.vstack(out_batch)  # type: ignore

        return out


class _TorchRegionLayer(TorchModule):
    """torch.nn.Module representing a RAT-SPN region layer

    Outputs are computed as a convex sum of the inputs where each output has its own weights.

    Args:
        num_nodes_region:
            Number of sum nodes in the region.
        partitions:
            List of child partition layers.
    """

    def __init__(
        self,
        children: List["_TorchPartitionLayer"],
        scope: List[int],
        num_nodes_region: int,
        weights: torch.Tensor = None,
    ) -> None:
        super(_TorchRegionLayer, self).__init__()

        self.scope = scope

        # register partitions as child modules
        for i, partition in enumerate(children, start=1):
            self.add_module(f"partition_{i}", partition)

        # compute number of inputs and outputs
        self.num_out = num_nodes_region
        self.num_in = sum(len(partition) for partition in self.children())

        if weights is None:
            # randomly generate weights (summing up to one per output)
            weights = torch.rand((self.num_out, self.num_in)) + 1e-08  # avoid zeros
            weights /= weights.sum(dim=-1, keepdim=True)

        # create and register (auxiliary) weight parameters for sum nodes
        self.register_parameter("weights_aux", Parameter())

        self.weights = weights

    @property
    def weights(self) -> torch.Tensor:
        # project auxiliary weights onto weights that sum up to one (per output)
        return proj_real_to_convex(self.weights_aux)

    @weights.setter
    def weights(self, values: torch.Tensor) -> None:
        if values.shape != (self.num_out, self.num_in):
            raise ValueError(
                f"Specified region weights for TorchRatSpn are of shape {values.shape}, but are expected to be of shape {(self.num_out, self.num_in)}."
            )
        elif not torch.allclose(values.sum(dim=-1), torch.tensor(1.0)):
            raise ValueError(
                "Region weights for TorchRatSpn must sum up to one for each output node."
            )

        self.weights_aux.data = proj_convex_to_real(values)

    def __len__(self) -> int:
        # return number of outputs
        return self.num_out

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:

        inputs = torch.hstack(inputs)  # type: ignore
        # broadcast inputs per output node and weight them in log-space
        weighted_inputs = inputs.unsqueeze(1) + self.weights.log()  # type: ignore

        return torch.logsumexp(weighted_inputs, dim=-1)


class _TorchLeafLayer(TorchModule):
    """torch.nn.Module representing a RAT-SPN leaf layer

    Contains a set of individual leaf nodes.

    Args:
        scope:
            List of integers representing the variable scope.
        num_nodes_leaf:
            Number of leaf nodes in the leaf layer.
    """

    def __init__(self, scope: List[int], num_nodes_leaf: int) -> None:
        super(_TorchLeafLayer, self).__init__()

        self.scope = scope

        # compute number of outputs
        self.num_out = num_nodes_leaf

        # TODO: set correct distributions
        # generate leaf nodes
        if(len(scope) == 1):
            self.leaf_nodes: List[TorchLeafNode] = [
                TorchGaussian(scope, 0.0, 1.0) for i in range(num_nodes_leaf)
            ]
        elif(len(scope) > 1):
            self.leaf_nodes: List[TorchLeafNode] = [
                TorchMultivariateGaussian(scope, torch.zeros(len(scope)), torch.eye(len(scope)))
            ]
        else:
            raise ValueError("Scope for _TorchLeafLayer of invalid size 0.")

        # register leaf nodes as child modules
        for i, leaf in enumerate(self.leaf_nodes, start=1):
            self.add_module(f"leaf_{i}", leaf)

    def __len__(self) -> int:
        # return number of outputs
        return self.num_out

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.hstack(inputs)  # type: ignore


class TorchRatSpn(TorchModule):
    """Torch backend module for RAT-SPNs.

    Args:
        region_graph:
            Region graph representing the high-level structure of the RAT-SPN.
        num_nodes_root:
            Number of nodes in the root region.
        num_nodes_region:
            Number of sum ndoes per internal region.
        num_nodes_leaf:
            Number of leaf nodes per leaf region.
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
        self.scope = (
            list(region_graph.root_region.random_variables) if region_graph.root_region else []
        )
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
            Union[Region, Partition],
            Union[_TorchRegionLayer, _TorchLeafLayer, _TorchPartitionLayer],
        ] = {}

        # recursively compute RatSpn layers beginning with root region
        self.root_region = _TorchRegionLayer(
            children=[
                self.partition(partition, list(region_graph.root_region.random_variables))
                for partition in region_graph.root_region.partitions
            ],
            scope=list(region_graph.root_region.random_variables),
            num_nodes_region=num_nodes_root,
        )

        # store root region mapping in dictionary
        self.rg_layers[region_graph.root_region] = self.root_region

        # randomly generate root node weights (summing up to one)
        root_node_weights = torch.rand((1, num_nodes_root)) + 1e-08  # avoid zeros
        root_node_weights /= root_node_weights.sum(dim=-1, keepdim=True)

        # create weights for root node
        self.register_parameter(
            "root_node_weights_aux",
            Parameter(),
        )

        self.root_node_weights = root_node_weights

    @property
    def root_node_weights(self) -> torch.Tensor:
        # project auxiliary weights onto weights that sum up to one (per output)
        return proj_real_to_convex(self.root_node_weights_aux)

    @root_node_weights.setter
    def root_node_weights(self, values: torch.Tensor) -> None:
        if values.shape != (1, self.num_nodes_root):
            raise ValueError(
                f"Specified root node weights for TorchRatSpn are of shape {values.shape}, but are expected to be of shape {(1, self.num_nodes_root)}."
            )
        elif not torch.allclose(values.sum(dim=-1), torch.tensor(1.0)):
            raise ValueError("Root node weights for TorchRatSpn must sum up to one.")

        self.root_node_weights_aux.data = proj_convex_to_real(values)

    def partition(self, partition: Partition, scope: List[int]) -> _TorchPartitionLayer:
        """Returns a _TorchPartitionLayer object from a region graph Partition.

        Args:
            partition:
                Region graph partition.
        """
        region_layers: List[Union[_TorchLeafLayer, _TorchRegionLayer]] = []

        # iterate over child regions
        for region in partition.regions:

            # leaf region (has no child partition)
            if not region.partitions:
                # create leaf layer with corresponding scope and number of leaf nodes
                leaf_layer = _TorchLeafLayer(scope=list(region.random_variables), num_nodes_leaf=self.num_nodes_leaf)  # type: ignore

                region_layers.append(leaf_layer)

                # store region mapping in dictionary
                self.rg_layers[region] = leaf_layer

            # internal region
            else:
                # create region layer and recursively convert child partitions
                region_layer = _TorchRegionLayer(
                    children=[
                        self.partition(p, list(region.random_variables)) for p in region.partitions
                    ],
                    scope=list(region.random_variables),
                    num_nodes_region=self.num_nodes_region,
                )

                region_layers.append(region_layer)

                # store region mapping in dictionary
                self.rg_layers[region] = region_layer

        # create partition layer with computed child region layers
        partition_layer = _TorchPartitionLayer(children=region_layers, scope=scope)

        # store partition mapping in dictionary
        self.rg_layers[partition] = partition_layer

        return partition_layer

    def __len__(self):
        # return number of outputs
        return 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # broadcast inputs per output node and weight them
        weighted_inputs = inputs.unsqueeze(1) + self.root_node_weights.log()  # type: ignore

        return torch.logsumexp(weighted_inputs, dim=-1)


@dispatch(_TorchRegionLayer, list)  # type: ignore[no-redef]
def _copy_region_parameters(src: _TorchRegionLayer, dst: List[ISumNode]) -> None:
    """Copy parameters from region layer to region sum nodes.

    Args:
        src:
            Region layer to copy parameters from.
        dst:
            List of region nodes to copy parameters to.
    """

    # number of sum nodes should match number of region layer outputs
    if not len(dst) == len(src):
        raise ValueError(
            "Number of ISumNodes and number of outputs for _TorchRegionLayer do not match."
        )

    # iterate over nodes and region weights
    for i, node in enumerate(dst):

        # number of children of sum nodes should match number of region layer inputs
        if not (len(node.weights) == len(node.children) == src.weights.data.shape[1]):
            raise ValueError(
                "Number of ISumNode children or weights and number of inputs for _TorchRegionLayer do not match."
            )

        # assign region weight slice to node weights
        node.weights = src.weights.data[i, :].detach().cpu().tolist()


@dispatch(_TorchLeafLayer, list)  # type: ignore[no-redef]
def _copy_region_parameters(src: _TorchLeafLayer, dst: List[ILeafNode]) -> None:
    """Copy parameters from leaf layer to leaf nodes.

    Args:
        src:
            Leaf layer to copy parameters from.
        dst:
            List of leaf nodes to copy parameters to.
    """

    # number of sum nodes should match number of region layer outputs
    if not len(dst) == len(src):
        raise ValueError(
            "Number of ILeafNodes and number of outputs for _TorchLeafLayer do not match."
        )

    # iterate over nodes and region weights
    for node, torch_node in zip(dst, src.leaf_nodes):

        # TODO: check whether node and torch node types match (e.g. call toNodes and check types)

        # copy leaf parameters from torch node
        node.set_params(*torch_node.get_params())


@dispatch(list, _TorchRegionLayer)  # type: ignore[no-redef]
def _copy_region_parameters(src: List[ISumNode], dst: _TorchRegionLayer) -> None:
    """Copy parameters from region sum nodes to region layer.

    Args:
        src:
            List of sum nodes to copy parameters from.
        dst:
            Region layer to copy parameters to.
    """

    # number of sum nodes should match number of region layer outputs
    if not len(dst) == len(src):
        raise ValueError(
            "Number of ISumNodes and number of outputs for _TorchRegionLayer do not match."
        )

    region_weights = dst.weights.detach()

    # iterate over nodes and region weights
    for i, node in enumerate(src):

        assert torch.isclose(
            torch.tensor(sum(node.weights)), torch.tensor(1.0)
        ), f"{torch.tensor(sum(node.weights))}"

        # number of children of sum nodes should match number of region layer inputs
        if not (len(node.weights) == len(node.children) == region_weights.shape[1]):
            raise ValueError(
                "Number of ISumNode children or weights and number of inputs for _TorchRegionLayer do not match."
            )

        # assign node weights to region weight slice
        region_weights.data[i, :] = torch.tensor(node.weights)

    dst.weights = region_weights


@dispatch(list, _TorchLeafLayer)  # type: ignore[no-redef]
def _copy_region_parameters(src: List[ILeafNode], dst: _TorchLeafLayer) -> None:
    """Copy parameters from leaf nodes to leaf layer.

    args:
        src:
            List of leaf nodes to copy parameters from.
        dst:
            Leaf layer to copy parameters to.
    """

    # there must be some sum nodes
    if len(dst) == 0:
        raise ValueError("No ISumNodes specified.")

    # number of sum nodes should match number of region layer outputs
    if not len(dst) == len(src):
        raise ValueError(
            "Number of ILeafNodes and number of outputs for _TorchLeafLayer do not match."
        )

    # iterate over nodes and region weights
    for node, torch_node in zip(src, dst.leaf_nodes):

        # TODO: check whether node and torch node types match (e.g. call toTorch and check types)

        # copy leaf parameters from node
        torch_node.set_params(*node.get_params())


@dispatch(TorchRatSpn)  # type: ignore[no-redef]
def toNodes(torch_rat: TorchRatSpn) -> RatSpn:
    # create RAT-SPN module using region graph (includes scopes)
    context = RandomVariableContext(
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
        region_layer: Union[
            _TorchRegionLayer, _TorchLeafLayer, _TorchPartitionLayer
        ] = torch_rat.rg_layers[region]

        # get region nodes from node RAT SPN
        region_nodes: List[INode] = rat.rg_nodes[region]

        # internal region
        if isinstance(region_layer, _TorchRegionLayer):
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
    rat.output_nodes[0].weights = torch_rat.root_node_weights.data.cpu().numpy()  # type: ignore

    return rat


@dispatch(RatSpn)  # type: ignore[no-redef]
def toTorch(rat: RatSpn) -> TorchRatSpn:
    torch_rat = TorchRatSpn(
        rat.region_graph, rat.num_nodes_root, rat.num_nodes_region, rat.num_nodes_leaf
    )

    # get all regions
    for region in rat.region_graph.regions:

        # get region layer from torch RAT SPN
        region_layer: Union[
            _TorchRegionLayer, _TorchLeafLayer, _TorchPartitionLayer
        ] = torch_rat.rg_layers[region]

        # get region nodes from node RAT SPN
        region_nodes: List[INode] = rat.rg_nodes[region]

        # internal region
        if isinstance(region_layer, _TorchRegionLayer):
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
    torch_rat.root_node_weights[0, :] = torch.tensor(rat.output_nodes[0].weights)

    return torch_rat
