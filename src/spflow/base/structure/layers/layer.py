# -*- coding: utf-8 -*-
"""Contains basic layer classes for SPFlow in the ``base`` backend.

Contains classes for layers of SPN-like sum- and product nodes.
"""
from typing import List, Union, Optional, Iterable
from copy import deepcopy

import numpy as np
import itertools

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.base.structure.module import Module, NestedModule
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode


class SPNSumLayer(NestedModule):
    r"""Layer representing multiple SPN-like sum nodes over all children in the ``base`` backend.

    Represents multiple convex combinations of its children over the same scope.

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        weights:
            Two-dimensional NumPy array containing non-negative weights for each input.
            Each row corresponds to a sum node with values summing up to one.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``SPNSumNode`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        n_nodes: int,
        children: List[Module],
        weights: Optional[
            Union[np.ndarray, List[List[float]], List[float]]
        ] = None,
        **kwargs,
    ) -> None:
        r"""Initializes ``SPNSumLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            children:
                Non-empty list of modules that are children to the layer.
                The output scopes for all child modules need to be equal.
            weights:
                Optional list of floats, list of lists of floats or one- to two-dimensional NumPy array,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized per row.

        Raises:
            ValueError: Invalid arguments.
        """
        if n_nodes < 1:
            raise ValueError(
                "Number of nodes for 'SPNSumLayer' must be greater of equal to 1."
            )

        if len(children) == 0:
            raise ValueError(
                "'SPNSumLayer' requires at least one child to be specified."
            )

        super(SPNSumLayer, self).__init__(children=children, **kwargs)

        self._n_out = n_nodes
        self.n_in = sum(child.n_out for child in self.children)

        # create input placeholder
        ph = self.create_placeholder(list(range(self.n_in)))

        # create sum nodes
        self.nodes = [SPNSumNode(children=[ph]) for _ in range(n_nodes)]

        # parse weights
        if weights is not None:
            self.weights = weights

        # compute scope
        self.scope = self.nodes[0].scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    @property
    def weights(self) -> np.ndarray:
        """Returns the weights of all nodes as a two-dimensional NumPy array."""
        return np.vstack([node.weights for node in self.nodes])

    @weights.setter
    def weights(
        self, values: Union[np.ndarray, List[List[float]], List[float]]
    ) -> None:
        """Sets the weights of all nodes to specified values.

        Args:
            values:
                List of floats, list of lists of floats or one- to two-dimensional NumPy array,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                Two-dimensional NumPy array containing non-negative weights for each input.

        Raises:
            ValueError: Invalid values.
        """
        if isinstance(values, list):
            values = np.array(values)
        if values.ndim != 1 and values.ndim != 2:
            raise ValueError(
                f"Numpy array of weight values for 'SPNSumLayer' is expected to be one- or two-dimensional, but is {values.ndim}-dimensional."
            )
        if not np.all(values > 0):
            raise ValueError("Weights for 'SPNSumLayer' must be all positive.")
        if not np.allclose(values.sum(axis=-1), 1.0):
            raise ValueError(
                "Weights for 'SPNSumLayer' must sum up to one in last dimension."
            )
        if not (values.shape[-1] == self.n_in):
            raise ValueError(
                "Number of weights for 'SPNSumLayer' in last dimension does not match total number of child outputs."
            )

        # same weights for all sum nodes
        if values.ndim == 1:
            for node in self.nodes:
                node.weights = values.copy()
        if values.ndim == 2:
            # same weights for all sum nodes
            if values.shape[0] == 1:
                for node in self.nodes:
                    node.weights = values.squeeze(0).copy()
            # different weights for all sum nodes
            elif values.shape[0] == self.n_out:
                for node, node_values in zip(self.nodes, values):
                    node.weights = node_values.copy()
            # incorrect number of specified weights
            else:
                raise ValueError(
                    f"Incorrect number of weights for 'SPNSumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {values.shape[0]}."
                )


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SPNSumLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SPNSumLayer, Module, None]:
    """Structural marginalization for SPN-like sum layer objects in the ``base`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.
    If the layer's scope is partially marginalized over, then a new sum layer over the marginalized child modules is returned.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        (Marginalized) sum layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute node scope (node only has single output)
    layer_scope = layer.scope

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if len(mutual_rvs) == len(layer_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        # TODO: pruning
        marg_children = []

        # marginalize child modules
        for child in layer.children:
            marg_child = marginalize(
                child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx
            )

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        return SPNSumLayer(
            n_nodes=layer.n_out, children=marg_children, weights=layer.weights
        )
    else:
        return deepcopy(layer)


class SPNProductLayer(NestedModule):
    r"""Layer representing multiple SPN-like product nodes over all children in the ``base`` backend.

    Represents multiple products of its children over pair-wise disjoint scopes.

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``SPNProductNode`` objects for the nodes in this layer.
    """

    def __init__(self, n_nodes: int, children: List[Module], **kwargs) -> None:
        r"""Initializes ``SPNProductLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            children:
                Non-empty list of modules that are children to the layer.
                The output scopes for all child modules need to be pair-wise disjoint.
        Raises:
            ValueError: Invalid arguments.
        """
        if n_nodes < 1:
            raise ValueError(
                "Number of nodes for 'SPNProductLayer' must be greater of equal to 1."
            )

        self._n_out = n_nodes

        if len(children) == 0:
            raise ValueError(
                "'SPNProductLayer' requires at least one child to be specified."
            )

        super(SPNProductLayer, self).__init__(children=children, **kwargs)

        # create input placeholder
        ph = self.create_placeholder(
            list(range(sum(child.n_out for child in self.children)))
        )
        # create prodcut nodes
        self.nodes = [SPNProductNode(children=[ph]) for _ in range(n_nodes)]

        self.scope = self.nodes[0].scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SPNProductLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SPNProductLayer, Module, None]:
    """Structural marginalization for SPN-like product layer objects in the ``base`` backend.

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
        (Marginalized) product layer or None if it is completely marginalized.
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

        marg_children = []

        # marginalize child modules
        for child in layer.children:
            marg_child = marginalize(
                child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx
            )

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        # if product node has only one child with a single ouput after marginalization and pruning is true, return child directly
        if len(marg_children) == 1 and marg_children[0].n_out == 1 and prune:
            return marg_children[0]
        else:
            return SPNProductLayer(n_nodes=layer.n_out, children=marg_children)
    else:
        return deepcopy(layer)


class SPNPartitionLayer(NestedModule):
    """Layer representing multiple SPN-like product nodes in the ``base`` backend as combinations of inputs from different partitions.

    A partition is a group of inputs over the same scope. Different partitions have pair-wise disjoint scopes.
    The layer represents all possible combinations of products selecting a single input from each partition.
    The resulting outputs all have the same scopes.

    Example:

        layer = SPNPartitionLayer([[node1, node2], [node3], [node4, node5, node6]])

        In this example the layer will have 2*1*3=6 product nodes over the following inputs (in this order):

            node1, node3, node4
            node1, node3, node5
            node1, node3, node6
            node2, node3, node4
            node2, node3, node5
            node2, node3, node6

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``SPNProductNode`` objects for the nodes in this layer.
        modules_per_partition:
            List of integers keeping track of the number of total inputs each input partition represents.
        partition_scopes:
            List of scopes keeping track of the scopes each partition represents.
    """

    def __init__(self, child_partitions: List[List[Module]], **kwargs) -> None:
        r"""Initializes ``SPNPartitionLayer`` object.

        Args:
            child_partitions:
                Non-empty list of lists of modules that are children to the layer.
                The output scopes for all child modules in a partition need to be qual.
                The output scopes for different partitions need to be pair-wise disjoint.
        Raises:
            ValueError: Invalid arguments.
        """
        if len(child_partitions) == 0:
            raise ValueError("No partitions for 'SPNPartitionLayer' specified.")

        scope = Scope()
        partition_sizes = []
        self.modules_per_partition = []
        self.partition_scopes = []

        # parse partitions
        for partition in child_partitions:
            # check if partition is empty
            if len(partition) == 0:
                raise ValueError(
                    "All partitions for 'SPNPartitionLayer' must be non-empty"
                )

            self.modules_per_partition.append(len(partition))
            partition_scope = Scope()
            size = 0

            # iterate over modules in this partition
            for child in partition:
                # increase total number of outputs of this partition
                size += child.n_out

                # for each output scope
                for s in child.scopes_out:
                    # check if query scope is the same
                    if (
                        partition_scope.equal_query(s)
                        or partition_scope.isempty()
                    ):
                        partition_scope = partition_scope.join(s)
                    else:
                        raise ValueError(
                            "Scopes of modules inside a partition must have same query scope."
                        )

            # add partition size to list
            partition_sizes.append(size)
            self.partition_scopes.append(partition_scope)

            # check if partition is pairwise disjoint to the overall scope so far (makes sure all partitions have pair-wise disjoint scopes)
            if partition_scope.isdisjoint(scope):
                scope = scope.join(partition_scope)
            else:
                raise ValueError(
                    "Scopes of partitions must be pair-wise disjoint."
                )

        super(SPNPartitionLayer, self).__init__(
            children=sum(child_partitions, []), **kwargs
        )

        self.n_in = sum(partition_sizes)
        self.nodes = []

        # create placeholders and nodes
        for input_ids in itertools.product(
            *np.split(list(range(self.n_in)), np.cumsum(partition_sizes[:-1]))
        ):
            ph = self.create_placeholder(input_ids)
            self.nodes.append(SPNProductNode(children=[ph]))

        self._n_out = len(self.nodes)
        self.scope = scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SPNPartitionLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SPNPartitionLayer, Module, None]:
    """Structural marginalization for SPN-like partition layer objects in the ``base`` backend.

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
        (Marginalized) partition layer or None if it is completely marginalized.
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
        partitions = np.split(
            children, np.cumsum(layer.modules_per_partition[:-1])
        )

        for partition_scope, partition_children in zip(
            layer.partition_scopes, partitions
        ):
            partition_children = partition_children.tolist()
            partition_mutual_rvs = set(partition_scope.query).intersection(
                set(marg_rvs)
            )

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
        if (
            len(marg_partitions) == 1
            and len(marg_partitions[0]) == 1
            and marg_partitions[0][0].n_out == 1
            and prune
        ):
            return marg_partitions[0][0]
        else:
            return SPNPartitionLayer(child_partitions=marg_partitions)
    else:
        return deepcopy(layer)


class SPNHadamardLayer(NestedModule):
    """Layer representing multiple SPN-like product nodes in the ``base`` backend as element-wise products of inputs from different partitions.

    A partition is a group of inputs over the same scope. Different partitions have pair-wise disjoint scopes.
    The layer represents element-wise products selecting a single input from each partition.
    This means that all partitions must represent an equal number of inputs or a single input (in which case the input is broadcast).
    The resulting outputs all have the same scopes.

    Example:

        layer = SPNHadamardLayer([[node1, node2], [node3], [node4, node5]])

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
            List of ``SPNProductNode`` objects for the nodes in this layer.
        modules_per_partition:
            List of integers keeping track of the number of total inputs each input partition represents.
        partition_scopes:
            List of scopes keeping track of the scopes each partition represents.
    """

    def __init__(self, child_partitions: List[List[Module]], **kwargs) -> None:
        r"""Initializes ``SPNHadamardLayer`` object.

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
            raise ValueError("No partitions for 'SPNHadamardLayer' specified.")

        scope = Scope()
        partition_sizes = []
        max_size = 1
        self.modules_per_partition = []
        self.partition_scopes = []

        # parse partitions
        for partition in child_partitions:
            # check if partition is empty
            if len(partition) == 0:
                raise ValueError(
                    "All partitions for 'SPNHadamardLayer' must be non-empty"
                )

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
                    if (
                        partition_scope.equal_query(s)
                        or partition_scope.isempty()
                    ):
                        partition_scope = partition_scope.join(s)
                    else:
                        raise ValueError(
                            "Scopes of modules inside a partition must have same query scope."
                        )

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
                raise ValueError(
                    "Scopes of partitions must be pair-wise disjoint."
                )

        super(SPNHadamardLayer, self).__init__(
            children=sum(child_partitions, []), **kwargs
        )

        self.n_in = sum(partition_sizes)
        self.nodes = []

        partition_indices = np.split(
            list(range(self.n_in)), np.cumsum(partition_sizes)[:-1]
        )

        # create placeholders and nodes
        for input_ids in zip(
            *[
                np.pad(indices, (0, max_size - size), mode="edge")
                for indices, size in zip(partition_indices, partition_sizes)
            ]
        ):
            ph = self.create_placeholder(list(input_ids))
            self.nodes.append(SPNProductNode(children=[ph]))

        self._n_out = len(self.nodes)
        self.scope = scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SPNHadamardLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SPNHadamardLayer, Module, None]:
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
        partitions = np.split(
            children, np.cumsum(layer.modules_per_partition[:-1])
        )

        for partition_scope, partition_children in zip(
            layer.partition_scopes, partitions
        ):
            partition_children = partition_children.tolist()
            partition_mutual_rvs = set(partition_scope.query).intersection(
                set(marg_rvs)
            )

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
        if (
            len(marg_partitions) == 1
            and len(marg_partitions[0]) == 1
            and marg_partitions[0][0].n_out == 1
            and prune
        ):
            return marg_partitions[0][0]
        else:
            return SPNHadamardLayer(child_partitions=marg_partitions)
    else:
        return deepcopy(layer)
