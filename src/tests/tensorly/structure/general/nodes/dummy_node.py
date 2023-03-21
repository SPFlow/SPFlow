from copy import deepcopy
from typing import Iterable, Optional, Union

from spflow.tensorly.structure import LeafNode
from spflow.meta.data import Scope
from spflow.meta.dispatch import (
    DispatchContext,
    dispatch,
    init_default_dispatch_context,
)


class DummyNode(LeafNode):
    """Dummy node class without children (to simulate leaf nodes)."""

    def __init__(self, scope: Optional[Scope] = None):

        if scope is None:
            scope = Scope([0])

        super().__init__(scope=scope)


@dispatch(memoize=True)
def marginalize(
    node: DummyNode,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, DummyNode]:

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
        return DummyNode(
            Scope(
                [rv for rv in node.scope.query if rv not in marg_rvs],
                node.scope.evidence,
            )
        )
    else:
        return deepcopy(node)
