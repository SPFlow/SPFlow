"""Lightweight training utilities for APC models."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.zoo.apc.config import ApcTrainConfig
from spflow.zoo.apc.model import AutoencodingPC


def _extract_batch_tensor(batch: Tensor | tuple | list) -> Tensor:
    """Extract the input tensor from a batch object.

    Accepts:
    - a plain tensor batch,
    - tuple/list batches where the first element is the input tensor.
    """
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) > 0 and isinstance(batch[0], Tensor):
        return batch[0]
    raise InvalidParameterError(
        f"Unsupported batch type {type(batch)}. Expected Tensor or tuple/list with Tensor as first element."
    )


def _to_loader(data: Tensor | Iterable, batch_size: int, shuffle: bool) -> Iterable:
    """Convert tensor data into a DataLoader, otherwise pass iterables through."""
    if isinstance(data, Tensor):
        return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=shuffle)
    return data


def _model_device(model: nn.Module) -> torch.device:
    """Infer the model device from parameters, defaulting to CPU for parameterless modules."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def train_apc_step(
    model: AutoencodingPC,
    batch: Tensor | tuple | list,
    optimizer: Optimizer,
    *,
    grad_clip_norm: float | None = None,
) -> dict[str, Tensor]:
    """Run a single APC optimization step.

    Args:
        model: APC model to optimize.
        batch: Input batch (tensor or tuple/list with tensor in index 0).
        optimizer: Optimizer instance.
        grad_clip_norm: Optional gradient clipping threshold.

    Returns:
        Detached tensor metrics produced by ``model.loss_components``.
    """
    del model, batch, optimizer, grad_clip_norm
    raise UnsupportedOperationError(
        "APC training helpers are unavailable after sample rollback."
    )


def evaluate_apc(
    model: AutoencodingPC,
    data: Tensor | Iterable,
    *,
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate mean APC losses on a dataset/iterator.

    Args:
        model: APC model to evaluate.
        data: Tensor dataset or iterable of batches.
        batch_size: Loader batch size when ``data`` is a tensor.

    Returns:
        Mean ``rec``, ``kld``, ``nll``, and ``total`` metrics.
    """
    del model, data, batch_size
    raise UnsupportedOperationError(
        "APC training helpers are unavailable after sample rollback."
    )


def fit_apc(
    model: AutoencodingPC,
    train_data: Tensor | Iterable,
    *,
    config: ApcTrainConfig,
    optimizer: Optimizer | None = None,
    val_data: Tensor | Iterable | None = None,
) -> list[dict[str, float]]:
    """Fit an APC model and return epoch-level metrics.

    Args:
        model: APC model to train.
        train_data: Tensor dataset or iterable of batches.
        config: Training hyperparameters.
        optimizer: Optional optimizer override. Defaults to Adam.
        val_data: Optional validation data source.

    Returns:
        Per-epoch metric dictionaries including train metrics and, when provided,
        validation metrics.
    """
    del model, train_data, config, optimizer, val_data
    raise UnsupportedOperationError(
        "APC training helpers are unavailable after sample rollback."
    )
