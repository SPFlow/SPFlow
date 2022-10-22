"""
Created on October 20, 2022

@authors: Philipp Deibert
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.leaves.parametric.cond_log_normal import CondLogNormal


@dispatch(memoize=True)
def log_likelihood(leaf: CondLogNormal, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve values for 'mean','std'
    mean, std = leaf.retrieve_params(data, dispatch_ctx)

    batch_size: int = data.shape[0]

    # get information relevant for the scope
    scope_data = data[:, leaf.scope.query]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, 1).to(mean.device)

    # ----- marginalization -----

    marg_ids = torch.isnan(scope_data).sum(dim=1) == len(leaf.scope.query)

    # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
    log_prob[marg_ids] = 0.0

    # ----- log probabilities -----

    # create masked based on distribution's support
    valid_ids = leaf.check_support(scope_data[~marg_ids]).squeeze(1)

    # TODO: suppress checks
    if not all(valid_ids):
        raise ValueError(
            f"Encountered data instances that are not in the support of the CondLogNormal distribution."
        )

    # compute probabilities for values inside distribution support
    log_prob[~marg_ids] = leaf.dist(mean=mean, std=std).log_prob(
        scope_data[~marg_ids].type(torch.get_default_dtype())
    )

    return log_prob