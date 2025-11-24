import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from spflow.modules.base import Module

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Track training and validation metrics during model training.

    Attributes:
        train_losses: List of training batch losses.
        val_losses: List of validation batch losses.
        train_correct: Number of correctly predicted training samples.
        train_total: Total number of training samples processed.
        val_correct: Number of correctly predicted validation samples.
        val_total: Total number of validation samples processed.
        training_steps: Total number of training batches processed.
        validation_steps: Total number of validation batches processed.
    """

    def __init__(self) -> None:
        """Initialize a new TrainingMetrics instance.

        All metrics are initialized to zero or empty lists.
        """
        self.train_losses: list[Tensor] = []
        self.val_losses: list[Tensor] = []
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
        self.training_steps = 0
        self.validation_steps = 0

    def update_train_batch(
        self, loss: Tensor, predicted: Tensor | None = None, targets: Tensor | None = None
    ) -> None:
        """Update metrics after processing a training batch.

        Args:
            loss: The computed loss for the batch.
            predicted: Predicted class labels (optional, for classification).
            targets: Ground truth target labels (optional, for classification).
        """
        self.train_losses.append(loss)
        self.training_steps += 1
        if predicted is not None and targets is not None:
            self.train_total += targets.size(0)
            self.train_correct += (predicted == targets).sum().item()

    def update_val_batch(
        self, loss: Tensor, predicted: Tensor | None = None, targets: Tensor | None = None
    ) -> None:
        """Update metrics after processing a validation batch.

        Args:
            loss: The computed loss for the batch.
            predicted: Predicted class labels (optional, for classification).
            targets: Ground truth target labels (optional, for classification).
        """
        self.val_losses.append(loss)
        self.validation_steps += 1
        if predicted is not None and targets is not None:
            self.val_total += targets.size(0)
            self.val_correct += (predicted == targets).sum().item()

    def get_train_accuracy(self) -> float:
        """Calculate training accuracy percentage.

        Returns:
            float: Training accuracy as a percentage (0-100). Returns 0.0 if
            no training samples have been processed.
        """
        return 100 * self.train_correct / self.train_total if self.train_total > 0 else 0.0

    def get_val_accuracy(self) -> float:
        """Calculate validation accuracy percentage.

        Returns:
            float: Validation accuracy as a percentage (0-100). Returns 0.0 if
            no validation samples have been processed.
        """
        return 100 * self.val_correct / self.val_total if self.val_total > 0 else 0.0

    def reset_epoch_metrics(self) -> None:
        """Reset all epoch-specific metrics."""
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0


def negative_log_likelihood_loss(model: Module, data: Tensor) -> torch.Tensor:
    """Compute negative log-likelihood loss.

    Args:
        model: Model to compute log-likelihood for.
        data: Input data tensor.

    Returns:
        torch.Tensor: Scalar negative log-likelihood loss tensor.
    """
    return -1 * model.log_likelihood(data).sum()


def nll_loss(ll: Tensor, target: Tensor) -> torch.Tensor:
    """Compute negative log-likelihood loss for classification tasks.

    Note:
        SPN models output log probabilities directly from their log_likelihood method,
        not raw logits like neural networks. Therefore, NLLLoss is the correct choice
        instead of CrossEntropyLoss. CrossEntropyLoss would apply log-softmax twice
        (once implicitly, once on already log-transformed probabilities), leading to
        incorrect results.

    Args:
        ll: Log-likelihood tensor with class probabilities.
        target: Target class labels as long tensor.

    Returns:
        torch.Tensor: Scalar negative log-likelihood loss tensor.
    """
    return nn.NLLLoss()(ll.squeeze(-1), target)


def _extract_batch_data(
    batch: tuple[Tensor, ...] | Tensor, is_classification: bool
) -> tuple[Tensor, Tensor | None]:
    """Extract data and targets from batch with proper error handling.

    Args:
        batch: Input batch from dataloader.
        is_classification: Whether this is a classification task.

    Returns:
        Tuple of (data, targets) where targets may be None for non-classification.

    Raises:
        ValueError: If batch format is invalid for the task type.
    """
    if is_classification:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("Classification batches must be (data, targets) tuples")
        return batch[0], batch[1]

    # Handle non-classification batch formats
    if isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            return batch[0], None
        elif len(batch) == 2:
            return batch[0], None  # Ignore second element
        else:
            raise ValueError("Non-classification batches should have 1 or 2 elements")
    else:
        return batch, None


def _process_training_batch(
    model: Module,
    batch: tuple[Tensor, ...] | Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    metrics: TrainingMetrics,
    is_classification: bool,
    callback_batch: Callable[[Tensor, int], None] | None,
) -> Tensor:
    """Process a single training batch and return the loss.

    Args:
        model: The model being trained.
        batch: Input batch from dataloader.
        optimizer: Optimizer for parameter updates.
        loss_fn: Loss function to compute.
        metrics: TrainingMetrics instance for tracking.
        is_classification: Whether this is a classification task.
        callback_batch: Optional callback function after each batch.

    Returns:
        The computed loss tensor.
    """
    # Clear gradients from previous step
    optimizer.zero_grad()
    data, targets = _extract_batch_data(batch, is_classification)

    # Compute loss based on task type (classification vs density estimation)
    if is_classification:
        #log_likelihood = model.log_likelihood(data)
        log_likelihood = model.predict_proba(data)
        loss = loss_fn(log_likelihood, targets) + -model.log_likelihood(data).mean()
        predicted = torch.argmax(log_likelihood, dim=-1).squeeze()
        metrics.update_train_batch(loss, predicted, targets)
    else:
        loss = loss_fn(model, data)
        metrics.update_train_batch(loss)

    # Backpropagate and update weights
    loss.backward()
    optimizer.step()

    if callback_batch is not None:
        callback_batch(loss, metrics.training_steps)

    return loss


def _run_validation_epoch(
    model: Module,
    validation_dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    metrics: TrainingMetrics,
    is_classification: bool,
    callback_batch: Callable[[Tensor, int], None] | None,
) -> Tensor:
    """Run validation epoch and return final validation loss.

    Args:
        model: The model being validated.
        validation_dataloader: DataLoader for validation data.
        loss_fn: Loss function to compute.
        metrics: TrainingMetrics instance for tracking.
        is_classification: Whether this is a classification task.
        callback_batch: Optional callback function after each batch.

    Returns:
        The final validation loss tensor from the last processed batch.
    """
    # Set model to evaluation mode
    model.eval()
    val_loss: Tensor

    # Validate without computing gradients
    with torch.no_grad():
        for batch in validation_dataloader:
            data, targets = _extract_batch_data(batch, is_classification)

            if is_classification:
                log_likelihood = model.log_likelihood(data)
                val_loss = loss_fn(log_likelihood, targets)
                predicted = torch.argmax(log_likelihood, dim=-1).squeeze()
                metrics.update_val_batch(val_loss, predicted, targets)
            else:
                val_loss = loss_fn(model, data)
                metrics.update_val_batch(val_loss)

            if callback_batch is not None:
                callback_batch(val_loss, metrics.training_steps)

    # Return to training mode
    model.train()
    return val_loss


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
        model: Model to train, must inherit from Module.
        dataloader: Training data loader yielding batches.
        epochs: Number of training epochs. Must be positive.
        verbose: Whether to log training progress per epoch.
        is_classification: Whether this is a classification task.
        optimizer: Optimizer instance. Defaults to Adam if None.
        scheduler: Learning rate scheduler. Defaults to MultiStepLR if None.
        lr: Learning rate for default Adam optimizer.
        loss_fn: Custom loss function. Defaults based on task type if None.
        validation_dataloader: Validation data loader for periodic evaluation.
        callback_batch: Function called after each batch with (loss, step).
        callback_epoch: Function called after each epoch with (losses, epoch).

    Raises:
        ValueError: If epochs is not a positive integer.
    """
    # Input validation
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    # Initialize components
    model.train()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1
        )

    # Initialize loss function based on task type
    if loss_fn is None:
        loss_fn = nll_loss if is_classification else negative_log_likelihood_loss
    metrics = TrainingMetrics()

    # Training loop
    for epoch in range(epochs):
        metrics.reset_epoch_metrics()

        # Process training batches
        for batch in dataloader:
            loss = _process_training_batch(
                model, batch, optimizer, loss_fn, metrics, is_classification, callback_batch
            )

        scheduler.step()

        # Log training metrics
        if is_classification:
            logger.debug(f"Accuracy: {metrics.get_train_accuracy():.2f}%")

        # Run validation
        if validation_dataloader is not None and epoch % 10 == 0:
            val_loss = _run_validation_epoch(
                model, validation_dataloader, loss_fn, metrics, is_classification, callback_batch
            )
            logger.debug(f"Validation Loss: {val_loss.item()}")
            if is_classification:
                logger.debug(f"Validation Accuracy: {metrics.get_val_accuracy():.2f}%")

        # Epoch callback and logging
        if callback_epoch is not None:
            callback_epoch(metrics.train_losses, epoch)

        if verbose:
            logger.info(f"Epoch [{epoch}/{epochs}]: Loss: {loss.item()}")
