# -*- coding: utf-8 -*-
"""Contains inference methods for RAT-SPNs for SPFlow in the ``base`` backend.
"""
import numpy as np
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.base.structure.rat.rat_spn import RatSPN


@dispatch(memoize=True)  # type: ignore
def log_likelihood(rat_spn: RatSPN, data: np.ndarray, check_support: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> np.ndarray:
    """Computes log-likelihoods for RAT-SPNs nodes in the ``base`` backend given input data.

    Args:
        sum_node:
            Sum node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return log_likelihood(rat_spn.root_node, data, check_support=check_support, dispatch_ctx=dispatch_ctx)