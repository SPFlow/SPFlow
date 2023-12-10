"""Contains SPN-like hadamard layer for SPFlow in the ``base`` backend.
"""
from copy import deepcopy
from typing import Iterable, List, Optional, Union
import numpy as np
import tensorly as tl
from ....utils.helper_functions import tl_split, tl_pad_edge, tl_tolist
from spflow.meta.structure import MetaModule
from spflow.tensorly.structure.module import Module
from spflow.tensorly.structure.nested_module import NestedModule
from spflow.tensorly.structure.spn.nodes.product_node import ProductNode

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class HadamardLayer(NestedModule):
    """Layer representing multiple SPN-like product nodes in the ``base`` backend as element-wise products of inputs from different partitions.

    A partition is a group of inputs over the same scope. Different partitions have pair-wise disjoint scopes.
    The layer represents element-wise products selecting a single input from each partition.
    This means that all partitions must represent an equal number of inputs or a single input (in which case the input is broadcast).
    The resulting outputs all have the same scopes.

    Example:

        layer = HadamardLayer([[node1, node2], [node3], [node4, node5]])

        In this example the layer will have 2 product nodes over the following inputs (in this order):

            node1, node3, node4
            node2, node3, node5

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``ProductNode`` objects for the nodes in this layer.
        modules_per_partition:
            List of integers keeping track of the number of total inputs each input partition represents.
        partition_scopes:
            List of scopes keeping track of the scopes each partition represents.
    """

    def __init__(self, child_partitions: List[List[MetaModule]], **kwargs) -> None:
        r"""Initializes ``HadamardLayer`` object.

        Args:
            child_partitions:
                Non-empty list of lists of modules that are children to the layer.
                The output scopes for all child modules in a partition need to be qual.
                The output scopes for different partitions need to be pair-wise disjoint.
                All partitions must have the same number of total outputs or a single output
                (in which case the output is broadcast).
        Raises:
            ValueError: Invalid arguments.
        """
        if len(child_partitions) == 0:
            raise ValueError("No partitions for 'HadamardLayer' specified.")

        scope = Scope()
        partition_sizes = []
        max_size = 1
        self.modules_per_partition = []
        self.partition_scopes = []

        # parse partitions
        for partition in child_partitions:
            # check if partition is empty
            if len(partition) == 0:
                raise ValueError("All partitions for 'HadamardLayer' must be non-empty")

            self.modules_per_partition.append(len(partition))
            partition_scope = Scope([])
            size = 0

            # iterate over modules in this partition
            for child in partition:
                # increase total number of outputs of this partition
                size += child.n_out

                # for each output scope
                for s in child.scopes_out:
                    # check if query scope is the same
                    if partition_scope.equal_query(s) or partition_scope.isempty():
                        partition_scope = partition_scope.join(s)
                    else:
                        raise ValueError("Scopes of modules inside a partition must have same query scope.")

            # add partition size to list
            if size == 1 or size == max_size or max_size == 1:
                # either max_size is 1, then set max size to size (greater or equal to 1) or max_size is greater than 1 in which case size must be max_size or 1
                max_size = max(size, max_size)
                partition_sizes.append(size)
            else:
                raise ValueError(
                    f"Total number of outputs per partition must be 1 or match the number of outputs of other partitions, but was {size}."
                )

            self.partition_scopes.append(partition_scope)

            # check if partition is pairwise disjoint to the overall scope so far (makes sure all partitions have pair-wise disjoint scopes)
            if partition_scope.isdisjoint(scope):
                scope = scope.join(partition_scope)
            else:
                raise ValueError("Scopes of partitions must be pair-wise disjoint.")

        super().__init__(children=sum(child_partitions, []), **kwargs)

        self.n_in = sum(partition_sizes)
        self.nodes = []

        partition_indices = np.split(list(range(self.n_in)), np.cumsum(partition_sizes[:-1])
)
        # create placeholders and nodes
        for input_ids in zip(
            *[
                #tl_pad_edge(indices, (0, max_size - size))
                #for indices, size in zip(partition_indices, partition_sizes)
                np.pad(indices, (0, max_size - size), mode="edge")
                for indices, size in zip(partition_indices, partition_sizes)
            ]
        ):
            ph = self.create_placeholder(list(input_ids))
            self.nodes.append(ProductNode(children=[ph]))

        self._n_out = len(self.nodes)
        #self.scope = scope
        self.scope = Scope([int(x) for x in scope.query], scope.evidence)
    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    def parameters(self):
        params = []
        for child in self.children:
            params.extend(list(child.parameters()))
        return params

    def to_dtype(self, dtype):
        self.dtype = dtype
        for node in self.nodes:
            node.dtype = dtype
        for child in self.children:
            child.to_dtype(dtype)

    def to_device(self, device):
        if self.backend == "numpy":
            raise ValueError("it is not possible to change the device of models that have a numpy backend")
        self.device = device
        for node in self.nodes:
            node.device = device
        for child in self.children:
            child.to_device(device)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: HadamardLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[HadamardLayer, MetaModule, None]:
    """Structural marginalization for SPN-like Hadamard layer objects in the ``base`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.
    If the layer's scope is partially marginalized over, then a new product layer over the marginalized child modules is returned.
    If the marginalized product layer has only one input and 'prune' is set, then the product node is pruned and the input is returned directly.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            If set to True and the marginalized node has a single input, the input is returned directly.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        (Marginalized) Hadamard layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute layer scope (same for all outputs)
    layer_scope = layer.scope

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # layer scope is being fully marginalized over
    if len(mutual_rvs) == len(layer_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        marg_partitions = []

        children = layer.children
        partitions = np.split(children, np.cumsum(layer.modules_per_partition[:-1]))

        for partition_scope, partition_children in zip(layer.partition_scopes, partitions):
            partition_children = partition_children.tolist()
            partition_mutual_rvs = set(partition_scope.query).intersection(set(marg_rvs))

            # partition scope is being fully marginalized over
            if len(partition_mutual_rvs) == len(partition_scope.query):
                # drop partition entirely
                continue
            # node scope is being partially marginalized
            elif partition_mutual_rvs:
                # marginalize child modules
                marg_partitions.append(
                    [
                        marginalize(
                            child,
                            marg_rvs,
                            prune=prune,
                            dispatch_ctx=dispatch_ctx,
                        )
                        for child in partition_children
                    ]
                )
            else:
                marg_partitions.append(deepcopy(partition_children))

        # if product node has only one input after marginalization and pruning is true, return input directly
        if len(marg_partitions) == 1 and len(marg_partitions[0]) == 1 and marg_partitions[0][0].n_out == 1 and prune:
            return marg_partitions[0][0]
        else:
            return HadamardLayer(child_partitions=marg_partitions)
    else:
        return deepcopy(layer)

@dispatch(memoize=True)  # type: ignore
def updateBackend(hadamard_layer: HadamardLayer, dispatch_ctx: Optional[DispatchContext] = None) -> HadamardLayer:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """

    children = hadamard_layer.children
    partitions = np.split(children, np.cumsum(hadamard_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return HadamardLayer(
        child_partitions=[
            [updateBackend(child, dispatch_ctx=dispatch_ctx) for child in partition] for partition in partitions
        ]
    )

@dispatch(memoize=True)  # type: ignore
def toNodeBased(hadamard_layer: HadamardLayer, dispatch_ctx: Optional[DispatchContext] = None) -> HadamardLayer:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """

    children = hadamard_layer.children
    partitions = np.split(children, np.cumsum(hadamard_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return HadamardLayer(
        child_partitions=[
            [toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in partition] for partition in partitions
        ]
    )

@dispatch(memoize=True)  # type: ignore
def toLayerBased(hadamard_layer: HadamardLayer, dispatch_ctx: Optional[DispatchContext] = None):
    from spflow.tensorly.structure.spn.layers_layerbased import HadamardLayer as HadamardLayerLayer
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """

    children = hadamard_layer.children
    partitions = np.split(children, np.cumsum(hadamard_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return HadamardLayerLayer(
        child_partitions=[
            [toLayerBased(child, dispatch_ctx=dispatch_ctx) for child in partition] for partition in partitions
        ]
    )
