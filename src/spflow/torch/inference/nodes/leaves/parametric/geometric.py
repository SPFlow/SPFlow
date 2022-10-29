# -*- coding: utf-8 -*-
"""Contains inference methods for ``Geometric`` nodes for SPFlow in the ``torch`` backend.
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.nodes.leaves.parametric.geometric import Geometric


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: Geometric,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    r"""Computes log-likelihoods for ``Geometric`` node in the ``torch`` backend given input data.

    Log-likelihood for ``Geometric`` is given by the logarithm of its probability distribution function (PDF):

    .. math::

        \log(\text{PMF}(k)) =  \log(p(1-p)^{k-1})

    where
        - :math:`k` is the number of trials
        - :math:`p` is the success probability of each trial

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf node to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the distribution.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.

    Raises:
        ValueError: Data outside of support.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # get information relevant for the scope
    scope_data = data[:, leaf.scope.query]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, 1).to(leaf.p.device)

    # ----- marginalization -----

    marg_ids = torch.isnan(scope_data).sum(dim=1) == len(leaf.scope.query)

    # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
    log_prob[marg_ids] = 0.0

    # ----- log probabilities -----

    if check_support:
        # create masked based on distribution's support
        valid_ids = leaf.check_support(scope_data[~marg_ids]).squeeze(1)

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the Geometric distribution."
            )

    # compute probabilities for values inside distribution support
    # data needs to be offset by -1 due to the different definitions between SciPy and PyTorch
    log_prob[~marg_ids] = leaf.dist.log_prob(
        scope_data[~marg_ids].type(torch.get_default_dtype()) - 1
    )

    return log_prob
