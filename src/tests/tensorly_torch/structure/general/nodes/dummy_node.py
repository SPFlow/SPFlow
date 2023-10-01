from copy import deepcopy
from typing import Iterable, Optional, Union

import torch
import tensorly as tl
import scipy.spatial.distance as np_dist
from spflow.meta.data import Scope
from spflow.meta.dispatch import (
    DispatchContext,
    dispatch,
    init_default_dispatch_context,
)
from spflow.tensorly.structure import LeafNode
from spflow.tensorly.utils.helper_functions import T


class DummyNode(LeafNode):
    """Dummy node class without children."""

    def __init__(self, scope: Optional[Scope] = None, loc=0.0):

        if scope is None:
            scope = Scope([0])

        self.loc = tl.tensor(loc)

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


class DummyLeaf(LeafNode):
    def __init__(self, scope: Optional[Scope] = None, loc=0.0):

        if scope is None:
            scope = Scope([0])

        self.loc = tl.tensor(loc)

        super().__init__(scope=scope)


@dispatch(memoize=True)
def log_likelihood(
    node: DummyLeaf,
    data: T,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> T:

    scope_data = data[:, node.scope.query]

    if tl.get_backend() == "numpy":
        dist = np_dist.cdist(scope_data, node.loc.reshape(1, 1), p=2)
    elif tl.get_backend() == "pytorch":
        dist = torch.cdist(scope_data, node.loc.reshape(1, 1), p=2)
    else:
        raise NotImplementedError("log_likelihood in dummy_node is not implemented for this backend")

    dist[dist <= 1.0] = 1.0
    dist[dist > 1.0] = 0.0

    return dist.log()


@dispatch(memoize=True)
def em(
    node: DummyLeaf,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:

    pass
