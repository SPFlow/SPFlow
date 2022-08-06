"""
Created on November 26, 2021

@authors: Philipp Deibert
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform


@dispatch(memoize=True)
def log_likelihood(leaf: Uniform, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    
    batch_size: int = data.shape[0]

    # get information relevant for the scope
    scope_data = data[:, leaf.scope.query]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, 1).to(leaf.start.device)

    # ----- marginalization -----

    marg_ids = torch.isnan(scope_data).sum(dim=1) == len(leaf.scope.query)

    # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
    log_prob[marg_ids] = 0.0

    # ----- log probabilities -----

    # create masked based on distribution's support
    valid_ids = leaf.check_support(scope_data[~marg_ids])

    # TODO: suppress checks
    if not all(valid_ids):
        raise ValueError(
            f"Encountered data instances that are not in the support of the TorchUniform distribution."
        )

    if leaf.support_outside:
        torch_valid_ids = torch.zeros(len(marg_ids), dtype=torch.bool)
        torch_valid_ids[~marg_ids] |= leaf.dist.support.check(scope_data[~marg_ids]).squeeze(1)
        
        # TODO: torch_valid_ids does not necessarily have the same dimension as marg_ids
        log_prob[~marg_ids & ~torch_valid_ids] = -float("inf")

        # compute probabilities for values inside distribution support
        log_prob[~marg_ids & torch_valid_ids] = leaf.dist.log_prob(
            scope_data[~marg_ids & torch_valid_ids].type(torch.get_default_dtype())
        )
    else:
        # compute probabilities for values inside distribution support
        log_prob[~marg_ids] = leaf.dist.log_prob(
            scope_data[~marg_ids].type(torch.get_default_dtype())
        )

    return log_prob