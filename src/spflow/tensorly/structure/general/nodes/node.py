"""Contains the basic abstract ``Node`` module that all nodes for SPFlow in the ``base`` backend.

All nodes in the ``base`` backend should inherit from ``Node`` or a subclass of it.
"""
from abc import ABC
from copy import deepcopy
from typing import Iterable, List, Optional, Union

from spflow.tensorly.structure.module import Module
from spflow.meta.structure import MetaModule
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class Node(Module, ABC):
    """Abstract base class for nodes in the ``base`` backend.

    All valid SPFlow node modules in the ``base`` backend should inherit from this class or a subclass of it.

    Attributes:
        children:
            List of modules that are children to the module in a directed graph.
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, children: Optional[List[MetaModule]] = None, **kwargs) -> None:
        r"""Initializes ``Node`` object.

        Initializes node by correctly setting its children.

        Args:
            children:
                Optional list of modules that are children to the node.
        """
        if children is None:
            children = []

        super().__init__(children=children, **kwargs)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this node. Returns one since nodes represent single outputs."""
        return 1

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this node represents."""
        return [self.scope]


@dispatch(memoize=True)  # type: ignore
def marginalize(
    node: Node,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[Node, None]:
    """Structural marginalization for node objects in the ``base`` backend.

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
    if len(mutual_rvs) == len(node_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        raise NotImplementedError(
            "Partial marginalization of 'Node' is not implemented for generic nodes. Dispatch an appropriate implementation for a specific node type."
        )
    else:
        return deepcopy(node)
