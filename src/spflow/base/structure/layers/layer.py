"""
Created on August 09, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable
from copy import deepcopy

import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module, NestedModule
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode


class SPNSumLayer(NestedModule):
    """Layer representing multiple SPN-like sum nodes over all children.

    Args:
        children: list of child modules (defaults to empty list).
    """
    def __init__(self, n: int, children: List[Module], weights: Optional[Union[np.ndarray, List[List[float]], List[float]]]=None, **kwargs) -> None:
        """TODO"""

        if(n < 1):
            raise ValueError("Number of nodes for 'SumLayer' must be greater of equal to 1.")

        if not children:
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
def marginalize(layer: SPNSumLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None):
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
        children: list of child modules (defaults to empty list).
    """
    def __init__(self, n: int, children: List[Module], **kwargs) -> None:
        """TODO"""

        if(n < 1):
            raise ValueError("Number of nodes for 'ProductLayer' must be greater of equal to 1.")

        self._n_out = n

        if not children:
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
def marginalize(layer: SPNProductLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None):
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