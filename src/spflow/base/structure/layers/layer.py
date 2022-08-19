"""
Created on August 09, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable
from copy import deepcopy

import numpy as np
import itertools

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module, NestedModule
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode


class SPNSumLayer(NestedModule):
    """Layer representing multiple SPN-like sum nodes over all children.

    Args:
        n: number of output nodes.
        children: list of child modules.
    """
    def __init__(self, n: int, children: List[Module], weights: Optional[Union[np.ndarray, List[List[float]], List[float]]]=None, **kwargs) -> None:
        """TODO"""

        if(n < 1):
            raise ValueError("Number of nodes for 'SumLayer' must be greater of equal to 1.")

        if len(children) == 0:
            raise ValueError("'SPNSumLayer' requires at least one child to be specified.")

        super(SPNSumLayer, self).__init__(children=children, **kwargs)

        self._n_out = n
        self.n_in = sum(child.n_out for child in self.children)

        # create input placeholder
        ph = self.create_placeholder(list(range(self.n_in)))

        # create sum nodes
        self.nodes = [SPNSumNode(children=[ph]) for _ in range(n)]

        # parse weights
        if(weights is not None):
            self.weights = weights

        # compute scope
        self.scope = self.nodes[0].scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out
    
    @property
    def scopes_out(self) -> List[Scope]:
        """TODO"""
        return [self.scope for _ in range(self.n_out)]

    @property
    def weights(self) -> np.ndarray:
        """TODO"""
        return np.vstack([node.weights for node in self.nodes])

    @weights.setter
    def weights(self, values: Union[np.ndarray, List[List[float]], List[float]]) -> None:
        """TODO"""
        if isinstance(values, list):
            values = np.array(values)
        if(values.ndim != 1 and values.ndim != 2):
            raise ValueError(f"Numpy array of weight values for 'SPNSumLayer' is expected to be one- or two-dimensional, but is {values.ndim}-dimensional.")
        if not np.all(values > 0):
            raise ValueError("Weights for 'SPNSumLayer' must be all positive.")
        if not np.allclose(values.sum(axis=-1), 1.0):
            raise ValueError("Weights for 'SPNSumLayer' must sum up to one in last dimension.")
        if not (values.shape[-1] == self.n_in):
            raise ValueError("Number of weights for 'SPNSumLayer' in last dimension does not match total number of child outputs.")
        
        # same weights for all sum nodes
        if(values.ndim == 1):
            for node in self.nodes:
                node.weights = values.copy()
        if(values.ndim == 2):
            # same weights for all sum nodes
            if(values.shape[0] == 1):
                for node in self.nodes:
                    node.weights = values.squeeze(0).copy()
            # different weights for all sum nodes
            elif(values.shape[0] == self.n_out):
                for node, node_values in zip(self.nodes, values):
                    node.weights = node_values.copy()
            # incorrect number of specified weights
            else:
                raise ValueError(f"Incorrect number of weights for 'SPNSumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {values.shape[0]}.")


@dispatch(memoize=True)
def marginalize(layer: SPNSumLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[SPNSumLayer, Module, None]:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute node scope (node only has single output)
    layer_scope = layer.scope

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if(len(mutual_rvs) == len(layer_scope.query)):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        # TODO: pruning
        marg_children = []

        # marginalize child modules
        for child in layer.children:
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)
        
        return SPNSumLayer(n=layer.n_out, children=marg_children, weights=layer.weights)
    else:
        return deepcopy(layer)


class SPNProductLayer(NestedModule):
    """Layer representing multiple SPN-like product nodes over all children.

    Args:
        n: number of output nodes.
        children: list of child modules.
    """
    def __init__(self, n: int, children: List[Module], **kwargs) -> None:
        """TODO"""

        if(n < 1):
            raise ValueError("Number of nodes for 'ProductLayer' must be greater of equal to 1.")

        self._n_out = n

        if len(children) == 0:
            raise ValueError("'SPNProductLayer' requires at least one child to be specified.")

        super(SPNProductLayer, self).__init__(children=children, **kwargs)
        
        # create input placeholder
        ph = self.create_placeholder(list(range(sum(child.n_out for child in self.children))))
        # create prodcut nodes
        self.nodes = [SPNProductNode(children=[ph]) for _ in range(n)]

        self.scope = self.nodes[0].scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out
    
    @property
    def scopes_out(self) -> List[Scope]:
        return [self.scope for _ in range(self.n_out)]


@dispatch(memoize=True)
def marginalize(layer: SPNProductLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[SPNProductLayer, Module, None]:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute layer scope (same for all outputs)
    layer_scope = layer.scope

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # layer scope is being fully marginalized over
    if(len(mutual_rvs) == len(layer_scope.query)):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:

        marg_children = []

        # marginalize child modules
        for child in layer.children:
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)
        
        # if product node has only one child after marginalization and pruning is true, return child directly
        if(len(marg_children) == 1 and prune):
            return marg_children[0]
        else:
            return SPNProductLayer(layer.n_out, children=marg_children)
    else:
        return deepcopy(layer)


class SPNPartitionLayer(NestedModule):
    """Layer representing multiple SPN-like product nodes partitions.

    Args:
        child_partitions: list of lists of child modules with pair-wise disoint scopes between partitions.
    """
    def __init__(self, child_partitions: List[List[Module]], **kwargs) -> None:
        """TODO"""

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
                raise ValueError("All partitions for 'SPNPartitionLayer' must be non-empty")
            
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
                    if partition_scope.equal_query(s) or partition_scope.isempty():
                        partition_scope = partition_scope.union(s)
                    else:
                        raise ValueError("Scopes of modules inside a partition must have same query scope.")

            # add partition size to list
            partition_sizes.append(size)
            self.partition_scopes.append(partition_scope)

            # check if partition is pairwise disjoint to the overall scope so far (makes sure all partitions have pair-wise disjoint scopes)
            if partition_scope.isdisjoint(scope):
                scope = scope.union(partition_scope)
            else:
                raise ValueError("Scopes of partitions must be pair-wise disjoint.")

        super(SPNPartitionLayer, self).__init__(children=sum(child_partitions, []), **kwargs)

        self.n_in = sum(partition_sizes)
        self.nodes = []
        
        # create placeholders and nodes
        for input_ids in itertools.product(*np.split(list(range(self.n_in)), np.cumsum(partition_sizes[:-1]))):
            ph = self.create_placeholder(input_ids)
            self.nodes.append(SPNProductNode(children=[ph]))
        
        self._n_out = len(self.nodes)
        self.scope = scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out
    
    @property
    def scopes_out(self) -> List[Scope]:
        return [self.scope for _ in range(self.n_out)]


@dispatch(memoize=True)
def marginalize(layer: SPNPartitionLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[SPNPartitionLayer, Module, None]:
    """TODO"""

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute layer scope (same for all outputs)
    layer_scope = layer.scope

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # layer scope is being fully marginalized over
    if(len(mutual_rvs) == len(layer_scope.query)):
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
            if(len(partition_mutual_rvs) == len(partition_scope.query)):
                # drop partition entirely
                continue
            # node scope is being partially marginalized
            elif partition_mutual_rvs:
                # marginalize child modules
                marg_partitions.append([marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx) for child in partition_children])
            else:
                marg_partitions.append(deepcopy(partition_children))

        # if product node has only one child after marginalization and pruning is true, return child directly
        if(len(marg_partitions) == 1 and len(marg_partitions[0]) == 1 and prune):
            return marg_partitions[0][0]
        else:
            return SPNPartitionLayer(child_partitions=marg_partitions)
    else:
        return deepcopy(layer)