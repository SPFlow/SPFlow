"""
Created on October 24, 2022

@authors: Philipp Deibert
"""
from abc import ABC
from typing import List, Union, Optional, Iterable, Callable
from copy import deepcopy

import torch
import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.cond_node import SPNCondSumNode as BaseSPNCondSumNode
from spflow.torch.structure.nodes.node import Node
from spflow.torch.structure.module import Module


class SPNCondSumNode(Node):
    """Torch version of a conditional sum node.

    Args:
        children: non-empty list of child modules.
        cond_f: TODO
    """
    def __init__(self, children: List[Module], cond_f: Optional[Callable]=None) -> None:
        """TODO"""
        super(SPNCondSumNode, self).__init__(children=children)

        if not children:
            raise ValueError("'SPNCondSumNode' requires at least one child to be specified.")
        
        scope = None

        for child in children:
            for s in child.scopes_out:
                if(scope is None):
                    scope = s
                else:
                    if not scope.equal_query(s):
                        raise ValueError(f"'SPNCondSumNode' requires child scopes to have the same query variables.")
                
                scope = scope.union(s)

        self.scope = scope
        self.n_in = sum(child.n_out for child in children)

        self.set_cond_f(cond_f)
    
    def set_cond_f(self, cond_f: Callable) -> None:
        self.cond_f = cond_f
    
    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> torch.Tensor:

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
        
        # if 'weights' was not already specified, retrieve it
        if weights is None:
            weights = cond_f(data)['weights']

        # if neither 'weights' nor 'cond_f' is specified (via node or arguments)
        if weights is None and cond_f is None:
            raise ValueError("'SPNCondSumNode' requires either 'weights' or 'cond_f' to retrieve 'weights' to be specified.")

        if isinstance(weights, list) or isinstance(weights, np.ndarray):
            weights = torch.tensor(weights).float()
        if(weights.ndim != 1):
            raise ValueError(f"Torch tensor of weight values for 'SPNCondSumNode' is expected to be one-dimensional, but is {weights.ndim}-dimensional.")
        if not torch.all(weights > 0):
            raise ValueError("Weights for 'SPNCondSumNode' must be all positive.")
        if not torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype)):
            raise ValueError("Weights for 'SPNCondSumNode' must sum up to one.")
        if not (len(weights) == self.n_in):
            raise ValueError("Number of weights for 'SPNCondSumNode' does not match total number of child outputs.")

        return weights


@dispatch(memoize=True)
def marginalize(sum_node: SPNCondSumNode, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None):
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute node scope (node only has single output)
    node_scope = sum_node.scope

    mutual_rvs = set(node_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if(len(mutual_rvs) == len(node_scope.query)):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        marg_children = []

        # marginalize child modules
        for child in sum_node.children():
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)
        
        return SPNCondSumNode(children=marg_children)
    else:
        return deepcopy(sum_node)


@dispatch(memoize=True)
def toBase(sum_node: SPNCondSumNode, dispatch_ctx: Optional[DispatchContext]=None) -> BaseSPNCondSumNode:
    """TODO"""
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseSPNCondSumNode(children=[toBase(child, dispatch_ctx=dispatch_ctx) for child in sum_node.children()])


@dispatch(memoize=True)
def toTorch(sum_node: BaseSPNCondSumNode, dispatch_ctx: Optional[DispatchContext]=None) -> SPNCondSumNode:
    """TODO"""
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SPNCondSumNode(children=[toTorch(child, dispatch_ctx=dispatch_ctx) for child in sum_node.children])