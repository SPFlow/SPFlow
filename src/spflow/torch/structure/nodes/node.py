# -*- coding: utf-8 -*-
"""Contains basic node classes for SPFlow in the ``torch`` backend.

Contains the abstract ``Node`` and ``LeafNode`` classes for SPFlow node modules in the ``torch`` backend
as well as classes for SPN-like sum- and product nodes.
"""
from abc import ABC
from typing import List, Union, Optional, Iterable
from copy import deepcopy

import torch
import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import SPNSumNode as BaseSPNSumNode
from spflow.base.structure.nodes.node import SPNProductNode as BaseSPNProductNode
from spflow.torch.structure.module import Module


# TODO: put projections somewhere else
def proj_convex_to_real(x: torch.Tensor) -> torch.Tensor:
    """TODO"""
    # convex coefficients are already normalized, so taking the log is sufficient
    return torch.log(x)


def proj_real_to_convex(x: torch.Tensor) -> torch.Tensor:
    """TODO"""
    return torch.nn.functional.softmax(x, dim=-1)


class Node(Module, ABC):
    """Abstract base class for nodes in the ``torch`` backend.

    All valid SPFlow node modules in the ``torch`` backend should inherit from this class or a subclass of it.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """
    def __init__(self, children: Optional[List[Module]]=None, **kwargs) -> None:
        r"""Initializes ``Node`` object.

        Initializes node by correctly setting its children.

        Args:
            children:
                Optional list of modules that are children to the node.
        """
        if(children is None):
            children = []

        super(Node, self).__init__(children=children, **kwargs)

    @property
    def n_out(self) -> int:
        """Returns the number of output for this node. Returns one since nodes represent single outputs."""
        return 1
    
    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this node represents."""
        return [self.scope]


@dispatch(memoize=True)  # type: ignore
def marginalize(node: Node, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[Node,None]:
    """Structural marginalization for node objects in the ``torch`` backend.

    Structurally marginalizes the specified node module.
    If the node's scope contains non of the random variables to marginalize, then the node is returned unaltered.
    If the node's scope is fully marginalized over, then None is returned.
    This implementation does not handle partial marginalization over the node's scope and instead raises an Error.

    Args:
        node:
            Node module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    
    Returns:
        Unaltered node if module is not marginalized or None if it is completely marginalized.

    Raises:
        ValueError: Partial marginalization of node's scope.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute node scope (node only has single output)
    node_scope = node.scope

    mutual_rvs = set(node_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if(len(mutual_rvs) == len(node_scope.query)):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        raise NotImplementedError("Partial marginalization of 'Node' is not implemented for generic nodes. Dispatch an appropriate implementation for a specific node type.")
    else:
        return deepcopy(node)


class SPNSumNode(Node):
    """SPN-like sum node in the ``torch`` backend.

    Represents a convex combination of its children over the same scope.
    Internally, the weights are represented as unbounded parameters that are projected onto convex combination for representing the actual weights.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        weights_aux:
            One-dimensional PyTorch tensor containing weights for each input.
        weights:
            One-dimensional PyTorch tensor containing non-negative weights for each input, summing up to one (projected from 'weights_aux').
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """
    def __init__(self, children: List[Module], weights: Optional[Union[np.ndarray, torch.Tensor, List[float]]]=None) -> None:
        """Initializes 'SPNSumNode' object.

        Args:
            children:
                Non-empty list of modules that are children to the node.
                The output scopes for all child modules need to be equal.
            weights:
                Optional list of floats, one-dimensional NumPy array or one-dimensional PyTorch tensor containing non-negative weights for each input, summing up to one.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized.

        Raises:
            ValueError: Invalid arguments.
        """
        super(SPNSumNode, self).__init__(children=children)

        if not children:
            raise ValueError("'SPNSumNode' requires at least one child to be specified.")
        
        scope = None

        for child in children:
            for s in child.scopes_out:
                if(scope is None):
                    scope = s
                else:
                    if not scope.equal_query(s):
                        raise ValueError(f"'SPNSumNode' requires child scopes to have the same query variables.")
                
                scope = scope.union(s)

        self.scope = scope
        self.n_in = sum(child.n_out for child in children)

        if weights is None:
            weights = torch.rand(self.n_in) + 1e-08  # avoid zeros
            weights /= weights.sum()

        # register auxiliary parameters for weights as torch parameters
        self.weights_aux = torch.nn.Parameter()
        # initialize weights
        self.weights = weights

    @property
    def weights(self) -> torch.Tensor:
        """Returns the weights of the node as a PyTorch tensor."""
        # project auxiliary weights onto weights that sum up to one
        return proj_real_to_convex(self.weights_aux)

    @weights.setter
    def weights(self, values: Union[np.ndarray, torch.Tensor, List[float]]) -> None:
        """Sets the weights of the node to specified values.

        Args:
            values:
                One-dimensional NumPy array, PyTorch tensor or list of floats of non-negative values summing up to one.
                Number of values must match number of total inputs to the node.

        Raises:
            ValueError: Invalid values.
        """
        if isinstance(values, list) or isinstance(values, np.ndarray):
            values = torch.tensor(values).float()
        if(values.ndim != 1):
            raise ValueError(f"Torch tensor of weight values for 'SPNSumNode' is expected to be one-dimensional, but is {values.ndim}-dimensional.")
        if not torch.all(values > 0):
            raise ValueError("Weights for 'SPNSumNode' must be all positive.")
        if not torch.isclose(values.sum(), torch.tensor(1.0, dtype=values.dtype)):
            raise ValueError("Weights for 'SPNSumNode' must sum up to one.")
        if not (len(values) == self.n_in):
            raise ValueError("Number of weights for 'SPNSumNode' does not match total number of child outputs.")

        self.weights_aux.data = proj_convex_to_real(values)


@dispatch(memoize=True)  # type: ignore
def marginalize(sum_node: SPNSumNode, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None):
    """Structural marginalization for ``SPNSumNode`` objects in the ``torch`` backend.

    Structurally marginalizes the specified sum node.
    If the sum node's scope contains non of the random variables to marginalize, then the node is returned unaltered.
    If the sum node's scope is fully marginalized over, then None is returned.
    If the sum node's scope is partially marginalized over, then a new sum node over the marginalized child modules is returned.

    Args:
        sum_node:
            Sum node module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect when marginalizing sum nodes. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    
    Returns:
        (Marginalized) sum node or None if it is completely marginalized.
    """
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
        
        return SPNSumNode(children=marg_children, weights=sum_node.weights)
    else:
        return deepcopy(sum_node)


@dispatch(memoize=True)  # type: ignore
def toBase(sum_node: SPNSumNode, dispatch_ctx: Optional[DispatchContext]=None) -> BaseSPNSumNode:
    """Conversion for ``SPNSumNode`` from ``torch`` backend to ``base`` backend.
    
    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseSPNSumNode(children=[toBase(child, dispatch_ctx=dispatch_ctx) for child in sum_node.children()], weights=sum_node.weights.detach().cpu().numpy())


@dispatch(memoize=True)  # type: ignore
def toTorch(sum_node: BaseSPNSumNode, dispatch_ctx: Optional[DispatchContext]=None) -> SPNSumNode:
    """Conversion for ``SPNSumNode`` from ``base`` backend to ``torch`` backend.
    
    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SPNSumNode(children=[toTorch(child, dispatch_ctx=dispatch_ctx) for child in sum_node.children], weights=sum_node.weights)


class SPNProductNode(Node):
    """SPN-like product node in the ``torch`` backend.

    Represents a product of its children over pair-wise disjoint scopes.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """
    def __init__(self, children: List[Module]) -> None:
        """Initializes ``SPNProductNode`` object.

        Args:
            children:
                Non-empty list of modules that are children to the node.
                The output scopes for all child modules need to be pair-wise disjoint.
        Raises:
            ValueError: Invalid arguments.
        """
        super(SPNProductNode, self).__init__(children=children)

        if not children:
            raise ValueError("'SPNProductNode' requires at least one child to be specified.")

        scope = Scope()

        for child in children:
            for s in child.scopes_out:
                if not scope.isdisjoint(s):
                    raise ValueError(f"'SPNProductNode' requires child scopes to be pair-wise disjoint.")

                scope = scope.union(s)

        self.scope = scope


@dispatch(memoize=True)  # type: ignore
def marginalize(product_node: SPNProductNode, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[Node,None]:
    """Structural marginalization for 'SPNProductNode' objects in the ``torch`` backend.

    Structurally marginalizes the specified product node.
    If the product node's scope contains non of the random variables to marginalize, then the node is returned unaltered.
    If the product node's scope is fully marginalized over, then None is returned.
    If the product node's scope is partially marginalized over, then a new prodcut node over the marginalized child modules is returned.
    If the marginalized product node has only one input and 'prune' is set, then the product node is pruned and the child is returned directly.

    Args:
        product_node:
            Sum node module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            If set to True and the marginalized node has a single input, the child is returned directly.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    
    Returns:
        (Marginalized) product node or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute node scope (node only has single output)
    node_scope = product_node.scope

    mutual_rvs = set(node_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if(len(mutual_rvs) == len(node_scope.query)):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        marg_children = []

        # marginalize child modules
        for child in product_node.children():
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)
        
        # if product node has only one child after marginalization and pruning is true, return child directly
        if(len(marg_children) == 1 and prune):
            return marg_children[0]
        else:
            return SPNProductNode(marg_children)
    else:
        return deepcopy(product_node)


@dispatch(memoize=True)  # type: ignore
def toBase(product_node: SPNProductNode, dispatch_ctx: Optional[DispatchContext]=None) -> BaseSPNProductNode:
    """Conversion for ``SPNProductNode`` from ``torch`` backend to ``base`` backend.
    
    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseSPNProductNode(children=[toBase(child, dispatch_ctx=dispatch_ctx) for child in product_node.children()])


@dispatch(memoize=True)  # type: ignore
def toTorch(product_node: BaseSPNProductNode, dispatch_ctx: Optional[DispatchContext]=None) -> SPNProductNode:
    """Conversion for ``SPNProductNode`` from ``base`` backend to ``torch`` backend.
    
    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SPNProductNode(children=[toTorch(child, dispatch_ctx=dispatch_ctx) for child in product_node.children])


class LeafNode(Node, ABC):
    """Abstract base class for leaf nodes in the ``torch`` backend.

    All valid SPFlow leaf nodes in the 'base' backend should inherit from this class or a subclass of it.
    
    Attributes:
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """
    def __init__(self, scope: Scope, **kwargs) -> None:
        """Initializes 'LeafNode' object.

        Args:
            scope:
                Scope object representing the scope of the leaf node,
        """
        super(LeafNode, self).__init__(children=[], **kwargs)

        self.scope = scope