"""Contains inference methods for ``CondCategoricalLayer`` leaves for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import numpy as np
import torch

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_categorical import (
    CondCategoricalLayer,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    layer: CondCategoricalLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    r"""Computes log-likelihoods for ``CondCategoricalLayer`` leaves in the ``torch`` backend given input data.

    Log-likelihood for ``CondCategoricalLayer`` is given by the logarithm of its individual probability mass functions (PMFs):

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

    Raises:
        ValueError: Data outside of support.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # retrieve value for 'p'
    k, p = layer.retrieve_params(data, dispatch_ctx)

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, layer.n_out).to(p.device)

    # query rvs of all node scopes
    query_rvs = [list(set(scope.query)) for scope in layer.scopes_out]

    # group nodes by equal scopes
    for query_signature in np.unique(query_rvs, axis=0):

        # compute all nodes with this scope
        node_ids = np.where((query_rvs == query_signature).all(axis=1))[0].tolist()
        node_ids_tensor = torch.tensor(node_ids)

        # get data for scope (since all "nodes" are univariate, order does not matter)
        scope_data = data[:, layer.scopes_out[node_ids[0]].query]

        # ----- marginalization -----
        marg_mask = torch.isnan(scope_data).sum(dim=1) == len(query_signature)
        marg_ids = torch.where(marg_mask)[0]
        non_marg_ids = torch.where(~marg_mask)[0]

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.meshgrid(marg_ids, node_ids_tensor, indexing="ij")] = 0.0

        # ----- log probabilities -----

        if check_support:
            # create masked based on distribution's support
            valid_ids = layer.check_support(data[~marg_mask], node_ids=node_ids, dispatch_ctx=dispatch_ctx)

            if not all(valid_ids.sum(dim=1)):
                raise ValueError(
                    f"Encountered data instances that are not in the support of the TorchCategorical distribution."
                )

        # compute probabilities for values inside distribution support
        log_prob[torch.meshgrid(non_marg_ids, node_ids_tensor, indexing="ij")] = layer.dist(
            p=p, node_ids=node_ids
        ).log_prob(scope_data[non_marg_ids, :].type(torch.get_default_dtype()))

    return log_prob
