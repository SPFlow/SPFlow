import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from spflow.exceptions import NumericalInstabilityError
from spflow.modules.module import Module
from spflow.utils.cache import Cache

logger = logging.getLogger(__name__)


def _check_finite_log_likelihood(log_likelihood: Tensor, context: str) -> None:
    """Raise if an EM log-likelihood is NaN or infinite.

    A NaN never satisfies the ``avg_ll <= prev_avg_ll`` convergence check, so without this guard EM would keep
    iterating (forever with ``max_steps=-1``) on a model whose parameters are already destroyed.

    Args:
        log_likelihood: Scalar (average or accumulated) log-likelihood.
        context: Location for the error message, e.g. ``"EM step 3"``.

    Raises:
        NumericalInstabilityError: If ``log_likelihood`` is not finite.
    """
    if not torch.isfinite(log_likelihood):
        raise NumericalInstabilityError(
            f"{context}: log-likelihood is {log_likelihood.item()}. Check leaf parameters for degenerate "
            "values (e.g. probabilities of exactly 0 or 1, zero scales) and the data for values outside the "
            "leaf support."
        )


def _retain_cached_log_likelihood_grads(cache: Cache) -> None:
    """Retain gradients for cached non-leaf likelihood tensors consumed by EM."""
    for lls in cache["log_likelihood"].values():
        if torch.is_tensor(lls) and lls.requires_grad:
            lls.retain_grad()


def _backward_accumulated_log_likelihood(acc_ll: Tensor) -> None:
    """Backpropagate one EM step without retaining the graph."""
    if not acc_ll.requires_grad:
        return

    acc_ll.backward()


def expectation_maximization(
    module: Module,
    data: Tensor,
    max_steps: int = -1,
    bias_correction: bool = True,
    verbose: bool = False,
) -> Tensor:
    """Performs expectation-maximization optimization on a given module.

    Args:
        module: Module to perform EM optimization on.
        data: Two-dimensional tensor containing the input data. Each row corresponds to a sample.
        max_steps: Maximum number of iterations. Defaults to -1, in which case optimization runs until convergence.
        bias_correction: Whether to apply bias correction in leaf MLE updates. Defaults to True.
        verbose: Whether to print the log-likelihood for each iteration step. Defaults to False.

    Returns:
        One-dimensional tensor containing the average log-likelihood for each iteration step.

    Raises:
        NumericalInstabilityError: If the log-likelihood becomes NaN or infinite.
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

        _check_finite_log_likelihood(avg_ll, f"EM step {step}")

        _retain_cached_log_likelihood_grads(cache)
        _backward_accumulated_log_likelihood(acc_ll)

        # recursively perform expectation maximization
        module._expectation_maximization_step(
            data=data,
            bias_correction=bias_correction,
            cache=cache,
        )

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
    bias_correction: bool = True,
    verbose: bool = False,
) -> Tensor:
    """Runs expectation-maximization over multiple epochs using mini-batches.

    Args:
        module: Module to perform EM optimization on.
        dataloader: Dataloader yielding batches of input data tensors.
        num_epochs: Number of epochs to iterate over the dataloader.
        bias_correction: Whether to apply bias correction in leaf MLE updates. Defaults to True.
        verbose: Whether to print the average log-likelihood per epoch.

    Returns:
        One-dimensional tensor containing the average log-likelihood for each epoch.

    Raises:
        NumericalInstabilityError: If the log-likelihood of a batch becomes NaN or infinite.
    """
    ll_history = []

    for epoch in range(num_epochs):
        epoch_ll = None
        num_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            batch_data = batch[0] if isinstance(batch, (list, tuple)) else batch
            cache = Cache()

            module_lls = module.log_likelihood(batch_data, cache=cache)
            acc_ll = module_lls.sum()
            _check_finite_log_likelihood(acc_ll.detach(), f"EM epoch {epoch}, batch {batch_idx}")

            if epoch_ll is None:
                epoch_ll = torch.zeros((), device=module_lls.device, dtype=module_lls.dtype)

            epoch_ll = epoch_ll + acc_ll.detach()
            num_samples += batch_data.shape[0]

            _retain_cached_log_likelihood_grads(cache)
            _backward_accumulated_log_likelihood(acc_ll)

            module._expectation_maximization_step(
                data=batch_data,
                bias_correction=bias_correction,
                cache=cache,
            )

        if epoch_ll is None or num_samples == 0:
            avg_ll = torch.tensor(float("nan"))
        else:
            avg_ll = epoch_ll / num_samples

        ll_history.append(avg_ll)

        if verbose:
            logger.info(f"Epoch {epoch}: Average log-likelihood: {avg_ll}")

    return torch.stack(ll_history)
