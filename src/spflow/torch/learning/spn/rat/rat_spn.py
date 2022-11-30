"""Contains learning methods for Random and Tensorized Sum-Product Networks (RAT-SPNs) for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.spn.rat.rat_spn import RatSPN


@dispatch(memoize=True)  # type: ignore
def em(
    rat_spn: RatSPN,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``RatSPN`` in the ``torch`` backend.

    Args:
        node:
            Node to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # call EM on root node
    em(
        rat_spn.root_node,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )
