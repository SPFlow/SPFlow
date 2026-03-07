"""Lightweight training utilities for APC models."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from spflow.exceptions import InvalidParameterError
from spflow.zoo.apc.config import ApcTrainConfig
from spflow.zoo.apc.model import AutoencodingPC


def _loader_generator() -> torch.Generator:
    """Create a shuffle generator compatible with the active torch default device."""
    get_default_device = getattr(torch, "get_default_device", None)
    default_device = torch.device(get_default_device()) if callable(get_default_device) else torch.device("cpu")
    generator_device = default_device.type if default_device.type != "meta" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(torch.initial_seed())
    return generator


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
        generator = _loader_generator() if shuffle else None
        return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=shuffle, generator=generator)
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
    model.train()
    x = _extract_batch_tensor(batch).to(_model_device(model))
    optimizer.zero_grad(set_to_none=True)
    metrics = model.loss_components(x)
    metrics["total"].backward()
    if grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    return {key: value.detach() for key, value in metrics.items() if isinstance(value, Tensor)}


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
    loader = _to_loader(data, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()

    totals = {"rec": 0.0, "kld": 0.0, "nll": 0.0, "total": 0.0}
    n_samples = 0
    device = _model_device(model)

    with torch.no_grad():
        for batch in loader:
            x = _extract_batch_tensor(batch).to(device)
            metrics = model.loss_components(x)
            batch_size_eff = int(x.shape[0])
            n_samples += batch_size_eff
            for key in totals:
                totals[key] += float(metrics[key].item()) * batch_size_eff

    if was_training:
        model.train()

    if n_samples <= 0:
        raise InvalidParameterError("Cannot evaluate APC on empty data.")

    return {key: value / n_samples for key, value in totals.items()}


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
    train_loader = _to_loader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = (
        None if val_data is None else _to_loader(val_data, batch_size=config.batch_size, shuffle=False)
    )

    if optimizer is None:
        optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        totals = {"rec": 0.0, "kld": 0.0, "nll": 0.0, "total": 0.0}
        n_samples = 0

        for batch in train_loader:
            metrics = train_apc_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                grad_clip_norm=config.grad_clip_norm,
            )
            x = _extract_batch_tensor(batch)
            batch_size_eff = int(x.shape[0])
            n_samples += batch_size_eff
            for key in totals:
                totals[key] += float(metrics[key].item()) * batch_size_eff

        if n_samples <= 0:
            raise InvalidParameterError("Cannot fit APC on empty training data.")

        row: dict[str, float] = {"epoch": float(epoch + 1)}
        row.update({f"train_{key}": value / n_samples for key, value in totals.items()})

        if val_loader is not None:
            val_metrics = evaluate_apc(model=model, data=val_loader, batch_size=config.batch_size)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})

        history.append(row)

    return history
