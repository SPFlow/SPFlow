"""Contains inference methods for ``CondCategorical`` nodes for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.cond_categorical import (
    CondCategorical,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: CondCategorical,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    r"""Computes log-likelihoods for ``CondCategorical`` node given input data in the ``torch`` backend.

    Log-likelihood for ``CondCategorical`` is given by the logarithm of its probability mass function (PMF):
    
    .. math::

        \log(\text{PMF}(k))=\log(p_k)

    where
        - :math:`k` is an integer associated with a category
        - :math:`p_k` is the success probability of the associated category k in :math:`[0,1]`

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
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # get information relevant for the scope
    scope_data = data[:, leaf.scope.query]

    # retrieve value for 'p'
    k, p = leaf.retrieve_params(data, dispatch_ctx)

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, 1).to(p.device)

    # ----- marginalization -----
    marg_mask = torch.isnan(scope_data).sum(dim=1) == len(leaf.scope.query)
    marg_ids = torch.where(marg_mask)[0]
    non_marg_ids = torch.where(~marg_mask)[0]

    # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
    log_prob[marg_ids] = 0.0

    # ----- log probabilities -----

    if check_support:
        # create masked based on distribution's support
        valid_ids = leaf.check_support(scope_data[non_marg_ids], is_scope_data=True, dispatch_ctx=dispatch_ctx).squeeze(1)

        if not torch.all(valid_ids):
            raise ValueError(f"Encountered data instances that are not in the support of the Categorical distribution: {valid_ids}")

    # compute probabilities for values inside distribution support
    print(p)
    log_prob[non_marg_ids] = leaf.dist(p=p).log_prob(scope_data[non_marg_ids, :].type(torch.get_default_dtype()))

    return log_prob
