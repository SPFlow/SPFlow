import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from spflow.modules.module import Module
from spflow.utils.cache import Cache

logger = logging.getLogger(__name__)


def expectation_maximization(
    module: Module,
    data: Tensor,
    max_steps: int = -1,
    verbose: bool = False,
) -> Tensor:
    """Performs expectation-maximization optimization on a given module.

    Args:
        module: Module to perform EM optimization on.
        data: Two-dimensional tensor containing the input data. Each row corresponds to a sample.
        max_steps: Maximum number of iterations. Defaults to -1, in which case optimization runs until convergence.
        verbose: Whether to print the log-likelihood for each iteration step. Defaults to False.

    Returns:
        One-dimensional tensor containing the average log-likelihood for each iteration step.
    """
    prev_avg_ll = torch.tensor(-float("inf"))
    ll_history = []

    if max_steps == -1:
        max_steps = 2**64 - 1

    for step in range(max_steps):
        # Shared cache for this EM iteration
        cache = Cache()

        # compute log likelihoods and sum them together
        module_lls = module.log_likelihood(data, cache=cache)
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


def expectation_maximization_batched(
    module: Module,
    dataloader: DataLoader,
    num_epochs: int = 1,
    verbose: bool = False,
) -> Tensor:
    """Runs expectation-maximization over multiple epochs using mini-batches.

    Args:
        module: Module to perform EM optimization on.
        dataloader: Dataloader yielding batches of input data tensors.
        num_epochs: Number of epochs to iterate over the dataloader.
        verbose: Whether to print the average log-likelihood per epoch.

    Returns:
        One-dimensional tensor containing the average log-likelihood for each epoch.
    """
    ll_history = []

    for epoch in range(num_epochs):
        epoch_ll = None
        num_samples = 0

        for batch in dataloader:
            batch_data = batch[0] if isinstance(batch, (list, tuple)) else batch
            cache = Cache()

            module_lls = module.log_likelihood(batch_data, cache=cache)
            acc_ll = module_lls.sum()

            if epoch_ll is None:
                epoch_ll = torch.zeros((), device=module_lls.device, dtype=module_lls.dtype)

            epoch_ll = epoch_ll + acc_ll.detach()
            num_samples += batch_data.shape[0]

            for lls in cache["log_likelihood"].values():
                if torch.is_tensor(lls) and lls.requires_grad:
                    lls.retain_grad()

            if acc_ll.requires_grad:
                acc_ll.backward(retain_graph=True)

            module.expectation_maximization(batch_data, cache=cache)

        if epoch_ll is None or num_samples == 0:
            avg_ll = torch.tensor(float("nan"))
        else:
            avg_ll = epoch_ll / num_samples

        ll_history.append(avg_ll)

        if verbose:
            logger.info(f"Epoch {epoch}: Average log-likelihood: {avg_ll}")

    return torch.stack(ll_history)
