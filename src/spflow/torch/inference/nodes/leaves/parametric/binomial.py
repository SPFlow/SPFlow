"""
Created on November 26, 2021

@authors: Philipp Deibert
"""

import torch
from multipledispatch import dispatch  # type: ignore
from typing import Dict
from spflow.base.memoize import memoize
from spflow.torch.structure.nodes.leaves.parametric import TorchBinomial


@dispatch(TorchBinomial, torch.Tensor, cache=dict)
@memoize(TorchBinomial)
def log_likelihood(leaf: TorchBinomial, data: torch.Tensor, cache: Dict) -> torch.Tensor:

    batch_size: int = data.shape[0]

    # get information relevant for the scope
    scope_data = data[:, list(leaf.scope)]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, 1)

    # ----- marginalization -----

    marg_ids = torch.isnan(scope_data).sum(dim=1) == len(leaf.scope)

    # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
    log_prob[marg_ids] = 0.0

    # ----- log probabilities -----

    # create masked based on distribution's support
    valid_ids = leaf.check_support(scope_data[~marg_ids])

    if not all(valid_ids):
        raise ValueError(
            f"Encountered data instances that are not in the support of the TorchBinomial distribution."
        )

    # compute probabilities for values inside distribution support
    log_prob[~marg_ids] = leaf.dist.log_prob(
        scope_data[~marg_ids].type(torch.get_default_dtype())
    )

    return log_prob