"""Contains the expectation maximization optimization parameter learner for SPFlow in the ``torch`` backend.
"""

import torch
from torch import Tensor

from spflow.meta.dispatch.dispatch_context import DispatchContext

# from spflow.torch.learning.general.node.leaf.bernoulli import em  # TODO

from spflow import em, log_likelihood
from spflow.modules.module import Module
import logging

logger = logging.getLogger(__name__)


def expectation_maximization(
    module: Module,
    data: Tensor,
    max_steps: int = -1,
    check_support: bool = True,
    verbose: bool = False,
) -> Tensor:
    """Performs partitioning usig randomized dependence coefficients (RDCs) to be used with the LearnSPN algorithm in the ``torch`` backend.

    Args:
        module:
            Module to perform EM optimization on.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        max_steps:
            Integer representing the maximum number of iterations.
            Defaults to -1, in which case the optimization is performed until convergence.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        verbose:
            Boolean value indicating whether or not to print the log-likelihood for each iteration step.
            Defaults to False.

    Returns:
        One-dimensional PyTorch tensors, containing the average log-likelihood for each iteration step.
    """
    prev_avg_ll = torch.tensor(-float("inf"))
    ll_history = []

    if max_steps == -1:
        max_steps = 2**64 - 1

    for step in range(max_steps):
        # initialize new dispatch context
        dispatch_ctx = DispatchContext()

        # compute log likelihoods and sum them together
        acc_ll = log_likelihood(module, data, check_support=check_support, dispatch_ctx=dispatch_ctx).sum()
        avg_ll = acc_ll.detach().clone() / data.shape[0]

        ll_history.append(avg_ll)

        if verbose:
            logger.info(f"Step {step}: Average log-likelihood: {avg_ll}")

        # retain gradients for all module log-likelihoods
        for lls in dispatch_ctx.cache["log_likelihood"].values():
            if lls.requires_grad:
                lls.retain_grad()

        # compute gradients (if there are differentiable parameters to begin with)
        if acc_ll.requires_grad:
            acc_ll.backward(retain_graph=True)

        # recursively perform expectation maximization
        em(module, data, check_support=check_support, dispatch_ctx=dispatch_ctx)

        # end update loop if max steps reached or loss converged
        if avg_ll <= prev_avg_ll:
            if verbose:
                logger.info(f"EM converged after {step} steps.")
            break

        prev_avg_ll = avg_ll
        # TODO: zero/None all gradients

    return torch.stack(ll_history)
