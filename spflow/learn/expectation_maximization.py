import logging

import torch
from torch import Tensor

from spflow.modules.base import Module

logger = logging.getLogger(__name__)


def expectation_maximization(
    module: Module,
    data: Tensor,
    max_steps: int = -1,
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
        verbose:
            Boolean value indicating whether to print the log-likelihood for each iteration step.
            Defaults to False.

    Returns:
        One-dimensional PyTorch tensors, containing the average log-likelihood for each iteration step.
    """
    prev_avg_ll = torch.tensor(-float("inf"))
    ll_history = []

    if max_steps == -1:
        max_steps = 2**64 - 1

    for step in range(max_steps):
        # Shared cache for this EM iteration
        cache = {"log_likelihood": {}}

        # compute log likelihoods and sum them together
        module_lls = module.log_likelihood(data, cache=cache)
        cache["log_likelihood"][module] = module_lls
        acc_ll = module_lls.sum()
        avg_ll = acc_ll.detach().clone() / data.shape[0]

        ll_history.append(avg_ll)

        if verbose:
            logger.info(f"Step {step}: Average log-likelihood: {avg_ll}")

        # retain gradients for all module log-likelihoods
        for lls in cache["log_likelihood"].values():
            if torch.is_tensor(lls) and lls.requires_grad:
                lls.retain_grad()

        # compute gradients (if there are differentiable parameters to begin with)
        if acc_ll.requires_grad:
            acc_ll.backward(retain_graph=True)

        # recursively perform expectation maximization
        module.expectation_maximization(data, cache=cache)

        # end update loop if max steps reached or loss converged
        if avg_ll <= prev_avg_ll:
            if verbose:
                logger.info(f"EM converged after {step} steps.")
            break

        prev_avg_ll = avg_ll

    return torch.stack(ll_history)
