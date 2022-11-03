# -*- coding: utf-8 -*-
"""Contains SPN-like product layer for SPFlow in the ``base`` backend.
"""
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.base.structure.module import Module
from spflow.base.structure.nested_module import NestedModule
from spflow.base.structure.spn.nodes.product_node import SPNProductNode

from typing import Optional, Iterable, Union, List
from copy import deepcopy


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
