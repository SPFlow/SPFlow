# -*- coding: utf-8 -*-
"""Contains ``ProductNode`` for SPFlow in the ``base`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.node import Node

from typing import List, Union, Optional, Iterable
from copy import deepcopy


class SPNProductNode(Node):
    r"""SPN-like product node in the ``base`` backend.

    Represents a product of its children over pair-wise disjoint scopes.

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, children: List[Module]) -> None:
        r"""Initializes ``SPNProductNode`` object.

        Args:
            children:
                Non-empty list of modules that are children to the node.
                The output scopes for all child modules need to be pair-wise disjoint.
        Raises:
            ValueError: Invalid arguments.
        """
        super(SPNProductNode, self).__init__(children=children)

        if not children:
            raise ValueError(
                "'SPNProductNode' requires at least one child to be specified."
            )

        scope = Scope()

        for child in children:
            for s in child.scopes_out:
                if not scope.isdisjoint(s):
                    raise ValueError(
                        f"'SPNProductNode' requires child scopes to be pair-wise disjoint."
                    )

                scope = scope.join(s)

        self.scope = scope


@dispatch(memoize=True)  # type: ignore
def marginalize(
    product_node: SPNProductNode,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SPNProductNode, Node, None]:
    r"""Structural marginalization for ``SPNProductNode`` objects in the ``base`` backend.

    Structurally marginalizes the specified product node.
    If the product node's scope contains non of the random variables to marginalize, then the node is returned unaltered.
    If the product node's scope is fully marginalized over, then None is returned.
    If the product node's scope is partially marginalized over, then a new prodcut node over the marginalized child modules is returned.
    If the marginalized product node has only one input and 'prune' is set, then the product node is pruned and the input is returned directly.

    Args:
        product_node:
            Sum node module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            If set to True and the marginalized node has a single input, the input is returned directly.
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
    if len(mutual_rvs) == len(node_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        marg_children = []

        # marginalize child modules
        for child in product_node.children:
            marg_child = marginalize(
                child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx
            )

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        # if product node has only one child with a single output after marginalization and pruning is true, return child directly
        if len(marg_children) == 1 and marg_children[0].n_out == 1 and prune:
            return marg_children[0]
        else:
            return SPNProductNode(marg_children)
    else:
        return deepcopy(product_node)
