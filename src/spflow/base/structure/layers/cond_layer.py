"""
Created on October 24, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable, Callable
from copy import deepcopy

import numpy as np
import itertools

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module, NestedModule
from spflow.base.structure.nodes.cond_node import SPNCondSumNode


class SPNCondSumLayer(NestedModule):
    """Layer representing multiple conditional SPN-like sum nodes over all children.

    Args:
        n: number of output nodes.
        children: list of child modules.
        cond_f: TODO
    """
    def __init__(self, n_nodes: int, children: List[Module], cond_f: Optional[Union[Callable,List[Callable]]]=None, **kwargs) -> None:
        """TODO"""

        if(n_nodes < 1):
            raise ValueError("Number of nodes for 'SPNCondSumLayer' must be greater of equal to 1.")

        if len(children) == 0:
            raise ValueError("'SPNCondSumLayer' requires at least one child to be specified.")

        super(SPNCondSumLayer, self).__init__(children=children, **kwargs)

        self._n_out = n_nodes
        self.n_in = sum(child.n_out for child in self.children)

        # create input placeholder
        ph = self.create_placeholder(list(range(self.n_in)))

        # create sum nodes
        self.nodes = [SPNCondSumNode(children=[ph]) for _ in range(n_nodes)]

        # compute scope
        self.scope = self.nodes[0].scope

        self.set_cond_f(cond_f)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out
    
    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]]=None) -> None:

        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError("'SPNCondSumLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes.")

        self.cond_f = cond_f
    
    @property
    def scopes_out(self) -> List[Scope]:
        """TODO"""
        return [self.scope for _ in range(self.n_out)]
    
    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> np.ndarray:
        """TODO"""
        weights, cond_f = None, None

        # check dispatch cache for required conditional parameter 'weights'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'weights' is specified (highest priority)
            if "weights" in args:
                weights = args["weights"]
            # check if alternative function to provide 'weights' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'weights' specified (lowest priority)
            cond_f = self.cond_f
        
        # if neither 'weights' nor 'cond_f' is specified (via node or arguments)
        if weights is None and cond_f is None:
            raise ValueError("'SPNCondSumLayer' requires either 'weights' or 'cond_f' to retrieve 'weights' to be specified.")

        # if 'weights' was not already specified, retrieve it
        if weights is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                weights = np.array([f(data)['weights'] for f in cond_f])
            else:
                weights = cond_f(data)['weights']

        if isinstance(weights, list):
            weights = np.array(weights)
        if(weights.ndim != 1 and weights.ndim != 2):
            raise ValueError(f"Numpy array of weight values for 'SPNCondSumLayer' is expected to be one- or two-dimensional, but is {weights.ndim}-dimensional.")
        if not np.all(weights > 0):
            raise ValueError("Weights for 'SPNCondSumLayer' must be all positive.")
        if not np.allclose(weights.sum(axis=-1), 1.0):
            raise ValueError("Weights for 'SPNCondSumLayer' must sum up to one in last dimension.")
        if not (weights.shape[-1] == self.n_in):
            raise ValueError("Number of weights for 'SPNCondSumLayer' in last dimension does not match total number of child outputs.")
        
        # same weights for all sum nodes
        if(weights.ndim == 1):
            # broadcast weights to all nodes
            weights = np.stack([weights for _ in range(self.n_out)])
        if(weights.ndim == 2):
            # same weights for all sum nodes
            if(weights.shape[0] == 1):
                # broadcast weights to all nodes
                weights = np.concatenate([weights for _ in range(self.n_out)], axis=0)
            # different weights for all sum nodes
            elif(weights.shape[0] == self.n_out):
                # already in correct output shape
                pass
            # incorrect number of specified weights
            else:
                raise ValueError(f"Incorrect number of weights for 'SPNCondSumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {weights.shape[0]}.")
        
        # TODO: check correct length of weights

        return weights

@dispatch(memoize=True)
def marginalize(layer: SPNCondSumLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[SPNCondSumLayer, Module, None]:
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
        
        return SPNCondSumLayer(n_nodes=layer.n_out, children=marg_children)
    else:
        return deepcopy(layer)