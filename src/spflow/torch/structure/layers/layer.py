# -*- coding: utf-8 -*-
"""Contains basic layer classes for SPFlow in the ``torch`` backend.

Contains classes for layers of SPN-like sum- and product nodes.
"""
from typing import List, Union, Optional, Iterable
from copy import deepcopy

import numpy as np
import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.torch.structure.module import Module
from spflow.torch.structure.nodes.node import (
    proj_real_to_convex,
    proj_convex_to_real,
)
from spflow.base.structure.layers.layer import SPNSumLayer as BaseSPNSumLayer
from spflow.base.structure.layers.layer import (
    SPNProductLayer as BaseSPNProductLayer,
)
from spflow.base.structure.layers.layer import (
    SPNPartitionLayer as BaseSPNPartitionLayer,
)
from spflow.base.structure.layers.layer import (
    SPNHadamardLayer as BaseSPNHadamardLayer,
)


class SPNSumLayer(Module):
    r"""Layer representing multiple SPN-like sum nodes over all children in the 'base' backend.

    Represents multiple convex combinations of its children over the same scope.
    Internally, the weights are represented as unbounded parameters that are projected onto convex combination for each node, representing the actual weights.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        weights_aux:
            Two-dimensional PyTorch tensor containing weights for each input and node.
            Each row corresponds to a node.
        weights:
            Two-dimensional PyTorch tensor containing non-negative weights for each input and node, summing up to one (projected from 'weights_aux').
            Each row corresponds to a node.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        n_nodes: int,
        children: List[Module],
        weights: Optional[
            Union[np.ndarray, torch.Tensor, List[List[float]], List[float]]
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
                Optional list of floats, list of lists of floats, one- to two-dimensional NumPy array or two-dimensional
                PyTorch tensor containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats, a one-dimensional
                NumPy array or a one-dimensonal PyTorch tensor, the same weights are reused for all sum nodes.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized per row.

        Raises:
            ValueError: Invalid arguments.
        """
        if n_nodes < 1:
            raise ValueError(
                "Number of nodes for 'SPNSumLayer' must be greater of equal to 1."
            )

        if not children:
            raise ValueError(
                "'SPNSumLayer' requires at least one child to be specified."
            )

        super(SPNSumLayer, self).__init__(children=children, **kwargs)

        self._n_out = n_nodes
        self.n_in = sum(child.n_out for child in self.children())

        # parse weights
        if weights is None:
            weights = torch.rand(self.n_out, self.n_in) + 1e-08  # avoid zeros
            weights /= weights.sum(dim=-1, keepdims=True)

        # register auxiliary parameters for weights as torch parameters
        self.weights_aux = torch.nn.Parameter()
        # initialize weights
        self.weights = weights

        # compute scope
        scope = None

        for child in children:
            for s in child.scopes_out:
                if scope is None:
                    scope = s
                else:
                    if not scope.equal_query(s):
                        raise ValueError(
                            f"'SPNSumLayer' requires child scopes to have the same query variables."
                        )

                scope = scope.union(s)

        self.scope = scope

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
        """Returns the weights of all nodes as a two-dimensional PyTorch tensor."""
        # project auxiliary weights onto weights that sum up to one
        return proj_real_to_convex(self.weights_aux)

    @weights.setter
    def weights(
        self,
        values: Union[np.ndarray, torch.Tensor, List[List[float]], List[float]],
    ) -> None:
        """Sets the weights of all nodes to specified values.

        Args:
            values:
                List of floats, list of lists of floats, one- to two-dimensional NumPy array or two-dimensional
                PyTorch tensor containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats, a one-dimensional
                NumPy array or a one-dimensonal PyTorch tensor, the same weights are reused for all sum nodes.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized per row.

        Raises:
            ValueError: Invalid values.
        """
        if isinstance(values, list) or isinstance(values, np.ndarray):
            values = torch.tensor(values).type(torch.get_default_dtype())
        if values.ndim != 1 and values.ndim != 2:
            raise ValueError(
                f"Torch tensor of weight values for 'SPNSumLayer' is expected to be one- or two-dimensional, but is {values.ndim}-dimensional."
            )
        if not torch.all(values > 0):
            raise ValueError("Weights for 'SPNSumLayer' must be all positive.")
        if not torch.allclose(values.sum(dim=-1), torch.tensor(1.0)):
            raise ValueError(
                "Weights for 'SPNSumLayer' must sum up to one in last dimension."
            )
        if not (values.shape[-1] == self.n_in):
            raise ValueError(
                "Number of weights for 'SPNSumLayer' in last dimension does not match total number of child outputs."
            )

        # same weights for all sum nodes
        if values.ndim == 1:
            self.weights_aux.data = proj_convex_to_real(
                values.repeat((self.n_out, 1)).clone()
            )
        if values.ndim == 2:
            # same weights for all sum nodes
            if values.shape[0] == 1:
                self.weights_aux.data = proj_convex_to_real(
                    values.repeat((self.n_out, 1)).clone()
                )
            # different weights for all sum nodes
            elif values.shape[0] == self.n_out:
                self.weights_aux.data = proj_convex_to_real(values.clone())
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
) -> Union[None, SPNSumLayer]:
    """Structural marginalization for SPN-like sum layer objects in the ``torch`` backend.

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
        for child in layer.children():
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


@dispatch(memoize=True)  # type: ignore
def toBase(
    sum_layer: SPNSumLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseSPNSumLayer:
    """Conversion for ``SPNSumLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseSPNSumLayer(
        n_nodes=sum_layer.n_out,
        children=[
            toBase(child, dispatch_ctx=dispatch_ctx)
            for child in sum_layer.children()
        ],
        weights=sum_layer.weights.detach().cpu().numpy(),
    )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    sum_layer: BaseSPNSumLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> SPNSumLayer:
    """Conversion for ``SPNSumLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        sum_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SPNSumLayer(
        n_nodes=sum_layer.n_out,
        children=[
            toTorch(child, dispatch_ctx=dispatch_ctx)
            for child in sum_layer.children
        ],
        weights=sum_layer.weights,
    )


class SPNProductLayer(Module):
    r"""Layer representing multiple SPN-like product nodes over all children in the ``torch`` backend.

    Represents multiple products of its children over pair-wise disjoint scopes.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
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

        if not children:
            raise ValueError(
                "'SPNProductLayer' requires at least one child to be specified."
            )

        super(SPNProductLayer, self).__init__(children=children, **kwargs)

        # compute scope
        scope = Scope()

        for child in children:
            for s in child.scopes_out:
                if not scope.isdisjoint(s):
                    raise ValueError(
                        f"'SPNProductNode' requires child scopes to be pair-wise disjoint."
                    )

                scope = scope.union(s)

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
    layer: SPNProductLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SPNProductLayer, Module, None]:
    """Structural marginalization for SPN-like product layer objects in the ``torch`` backend.

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
        for child in layer.children():
            marg_child = marginalize(
                child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx
            )

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        # if product node has only one child after marginalization and pruning is true, return child directly
        if len(marg_children) == 1 and prune:
            return marg_children[0]
        else:
            return SPNProductLayer(n_nodes=layer.n_out, children=marg_children)
    else:
        return deepcopy(layer)


@dispatch(memoize=True)  # type: ignore
def toBase(
    product_layer: SPNProductLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseSPNProductLayer:
    """Conversion for ``SPNProductLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        product_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseSPNProductLayer(
        n_nodes=product_layer.n_out,
        children=[
            toBase(child, dispatch_ctx=dispatch_ctx)
            for child in product_layer.children()
        ],
    )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    product_layer: BaseSPNProductLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> SPNProductLayer:
    """Conversion for ``SPNProductLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        product_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SPNProductLayer(
        n_nodes=product_layer.n_out,
        children=[
            toTorch(child, dispatch_ctx=dispatch_ctx)
            for child in product_layer.children
        ],
    )


class SPNPartitionLayer(Module):
    """Layer representing multiple SPN-like product nodes in the ``torch`` backend as combinations of inputs from different partitions.

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

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
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
        self.partition_sizes = []
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
                        partition_scope = partition_scope.union(s)
                    else:
                        raise ValueError(
                            "Scopes of modules inside a partition must have same query scope."
                        )

            # add partition size to list
            self.partition_sizes.append(size)
            self.partition_scopes.append(partition_scope)

            # check if partition is pairwise disjoint to the overall scope so far (makes sure all partitions have pair-wise disjoint scopes)
            if partition_scope.isdisjoint(scope):
                scope = scope.union(partition_scope)
            else:
                raise ValueError(
                    "Scopes of partitions must be pair-wise disjoint."
                )

        super(SPNPartitionLayer, self).__init__(
            children=sum(child_partitions, []), **kwargs
        )

        self.n_in = sum(self.partition_sizes)
        self._n_out = torch.prod(torch.tensor(self.partition_sizes)).item()
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
    """Structural marginalization for SPN-like partition layer objects in the ``torch`` backend.

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

        children = list(layer.children())
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

        # if product node has only one child after marginalization and pruning is true, return child directly
        if len(marg_partitions) == 1 and len(marg_partitions[0]) == 1 and prune:
            return marg_partitions[0][0]
        else:
            return SPNPartitionLayer(child_partitions=marg_partitions)
    else:
        return deepcopy(layer)


@dispatch(memoize=True)  # type: ignore
def toBase(
    partition_layer: SPNPartitionLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseSPNPartitionLayer:
    """Conversion for ``SPNPartitionLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        partition_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    children = list(partition_layer.children())
    partitions = np.split(
        children, np.cumsum(partition_layer.modules_per_partition[:-1])
    )

    return BaseSPNPartitionLayer(
        child_partitions=[
            [toBase(child, dispatch_ctx=dispatch_ctx) for child in partition]
            for partition in partitions
        ]
    )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    partition_layer: BaseSPNPartitionLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> SPNPartitionLayer:
    """Conversion for ``SPNPartitionLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        partition_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    children = list(partition_layer.children)
    partitions = np.split(
        children, np.cumsum(partition_layer.modules_per_partition[:-1])
    )

    return SPNPartitionLayer(
        child_partitions=[
            [toTorch(child, dispatch_ctx=dispatch_ctx) for child in partition]
            for partition in partitions
        ]
    )


class SPNHadamardLayer(Module):
    """Layer representing multiple SPN-like product nodes in the ``torch`` backend as element-wise products of inputs from different partitions.

    A partition is a group of inputs over the same scope. Different partitions have pair-wise disjoint scopes.
    The layer represents element-wise products selecting a single input from each partition.
    This means that all partitions must represent an equal number of inputs or a single input (in which case the input is broadcast).
    The resulting outputs all have the same scopes.

    Example:

        layer = SPNHadamardLayer([[node1, node2], [node3], [node4, node5]])

        In this example the layer will have 2 product nodes over the following inputs (in this order):

            node1, node3, node4
            node2, node3, node5

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
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
        max_size = 1
        self.partition_sizes = []
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
                        partition_scope = partition_scope.union(s)
                    else:
                        raise ValueError(
                            "Scopes of modules inside a partition must have same query scope."
                        )

            # add partition size to list
            if size == 1 or size == max_size or max_size == 1:
                # either max_size is 1, then set max size to size (greater or equal to 1) or max_size is greater than 1 in which case size must be max_size or 1
                max_size = max(size, max_size)
                self.partition_sizes.append(size)
            else:
                raise ValueError(
                    f"Total number of outputs per partition must be 1 or match the number of outputs of other partitions, but was {size}."
                )

            self.partition_scopes.append(partition_scope)

            # check if partition is pairwise disjoint to the overall scope so far (makes sure all partitions have pair-wise disjoint scopes)
            if partition_scope.isdisjoint(scope):
                scope = scope.union(partition_scope)
            else:
                raise ValueError(
                    "Scopes of partitions must be pair-wise disjoint."
                )

        super(SPNHadamardLayer, self).__init__(
            children=sum(child_partitions, []), **kwargs
        )

        self.n_in = sum(self.partition_sizes)
        self._n_out = max_size
        self.scope = scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]


@dispatch(memoize=True)  # typ: ignore
def marginalize(
    layer: SPNHadamardLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SPNHadamardLayer, Module, None]:
    """Structural marginalization for SPN-like Hadamard layer objects in the ``torch`` backend.

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

        children = list(layer.children())
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

        # if product node has only one child after marginalization and pruning is true, return child directly
        if len(marg_partitions) == 1 and len(marg_partitions[0]) == 1 and prune:
            return marg_partitions[0][0]
        else:
            return SPNHadamardLayer(child_partitions=marg_partitions)
    else:
        return deepcopy(layer)


@dispatch(memoize=True)  # type: ignore
def toBase(
    hadamard_layer: SPNHadamardLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseSPNHadamardLayer:
    """Conversion for ``SPNHadamardLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        hadamard_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    children = list(hadamard_layer.children())
    partitions = np.split(
        children, np.cumsum(hadamard_layer.modules_per_partition[:-1])
    )

    return BaseSPNHadamardLayer(
        child_partitions=[
            [toBase(child, dispatch_ctx=dispatch_ctx) for child in partition]
            for partition in partitions
        ]
    )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    hadamard_layer: BaseSPNHadamardLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> SPNHadamardLayer:
    """Conversion for ``SPNHadamardLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        hadamard_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    children = list(hadamard_layer.children)
    partitions = np.split(
        children, np.cumsum(hadamard_layer.modules_per_partition[:-1])
    )

    return SPNHadamardLayer(
        child_partitions=[
            [toTorch(child, dispatch_ctx=dispatch_ctx) for child in partition]
            for partition in partitions
        ]
    )
