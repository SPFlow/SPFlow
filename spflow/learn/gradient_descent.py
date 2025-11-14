import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from spflow.modules.base import Module

logger = logging.getLogger(__name__)


def negative_log_likelihood_loss(model: Module, data: Tensor) -> torch.Tensor:
    """Compute negative log-likelihood loss.

    Args:
        model: Model to compute log-likelihood for.
        data: Input data tensor.

    Returns:
        Negative log-likelihood loss tensor.
    """
    return -1 * model.log_likelihood(data).sum()


def nll_loss(ll: Tensor, target: Tensor) -> torch.Tensor:
    """Compute NLL loss for classification.

    Args:
        ll: Log-likelihood tensor.
        target: Target class labels.

    Returns:
        Negative log-likelihood loss tensor.
    """
    return nn.NLLLoss()(ll.squeeze(1), target)


def train_gradient_descent(
    model: Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = -1,
    verbose: bool = False,
    is_classification: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    lr: float = 1e-3,
    loss_fn: Callable[[Module, Tensor], Tensor] | None = None,
    validation_dataloader: torch.utils.data.DataLoader | None = None,
    callback_batch: Callable[[Tensor, int], None] | None = None,
    callback_epoch: Callable[[list[Tensor], int], None] | None = None,
):
    """Train model using gradient descent.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        epochs: Number of training epochs.
        verbose: Print loss per epoch.
        is_classification: Classification task flag.
        optimizer: Optimizer instance (defaults to Adam).
        scheduler: LR scheduler (defaults to MultiStepLR).
        lr: Learning rate if optimizer not provided.
        loss_fn: Loss function (defaults based on task type).
        validation_dataloader: Validation data loader.
        callback_batch: Callback after each batch.
        callback_epoch: Callback after each epoch.
    """
    model.train()

    # If no optimizer is provided, use Adam by default
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1
        )

    if loss_fn is None:
        if is_classification:
            loss_fn = nll_loss
        else:
            loss_fn = negative_log_likelihood_loss

    steps = 0
    for epoch in range(epochs):
        # Collect losses for each epoch

        # counter for accuracy
        correct = 0
        total = 0

        losses_epoch = []
        for batch in dataloader:
            # Reset gradients
            optimizer.zero_grad()

            # Compute negative log likelihood
            if is_classification:
                data, y = batch
                ll = model.log_likelihood(data)
                loss = loss_fn(ll, y)
                predicted = torch.argmax(ll, dim=-1).squeeze()

                total += y.size(0)
                correct += (predicted == y).sum().item()
            else:
                if isinstance(batch, tuple):
                    if len(batch) == 1:
                        data = batch[0]
                    elif len(batch) == 2:
                        data, _ = batch
                    else:
                        raise ValueError("Batch should be a tuple of length 1 or 2")

                else:
                    data = batch[0]

                loss = loss_fn(model, data)

            # Collect loss
            losses_epoch.append(loss)

            # Compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            # Count global steps
            steps += 1

            # Call callback function after each batch
            if callback_batch is not None:
                callback_batch(loss, steps)
        scheduler.step()
        # print(f"Epoch [{epoch}/{epochs}]: Loss: {loss.item()/dataloader.batch_size}")
        # print(f"Epoch [{epoch}/{epochs}]: Loss: {loss.item()}")
        if is_classification:
            logger.debug(f"Accuracy: {100 * correct / total}")

        # Call callback function after each epoch
        if callback_epoch is not None:
            callback_epoch(losses_epoch, epoch)

        if verbose:
            logger.info(f"Epoch [{epoch}/{epochs}]: Loss: {loss.item()}")

        if validation_dataloader is not None and epoch % 10 == 0:
            model.eval()
            total_val = 0
            correct_val = 0
            with torch.no_grad():
                for data, y in validation_dataloader:
                    if is_classification:
                        ll = model.log_likelihood(data)
                        val_loss = loss_fn(ll, y)
                        predicted = torch.argmax(ll, dim=-1).squeeze()

                        total_val += y.size(0)
                        correct_val += (predicted == y).sum().item()
                    else:
                        val_loss = loss_fn(model, data)

                    # Count global steps
                    steps += 1

                    # Call callback function after each batch
                    if callback_batch is not None:
                        callback_batch(val_loss, steps)
            logger.debug(f"Validation Loss: {val_loss.item()}")
            if is_classification:
                logger.debug(f"Validation Accuracy: {100 * correct_val / total_val}")
            model.train()
