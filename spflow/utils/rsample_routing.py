from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from spflow.utils.diff_sampling import (
    DiffSampleMethod,
    sample_categorical_differentiably,
    select_with_soft_or_hard,
)
from spflow.utils.diff_sampling_context import DifferentiableSamplingContext
from spflow.utils.sampling_context import SamplingContext


def _as_method(method: str | DiffSampleMethod) -> DiffSampleMethod:
    if isinstance(method, DiffSampleMethod):
        return method
    return DiffSampleMethod(method)


def _align_feature_axis(tensor: Tensor, target_features: int) -> Tensor:
    if tensor.shape[1] == target_features:
        return tensor
    if tensor.shape[1] == 1:
        expand_shape = (-1, target_features) + (-1,) * (tensor.ndim - 2)
        return tensor.expand(*expand_shape)
    if tensor.shape[1] > target_features:
        return tensor[:, :target_features, ...]
    head = tensor[:, :1, ...]
    expand_shape = (-1, target_features) + (-1,) * (tensor.ndim - 2)
    return head.expand(*expand_shape)


def _selector_for_dim(selector: Tensor, tensor: Tensor, dim: int) -> Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if selector.ndim == 2:
        selector = selector.unsqueeze(1)
    selector = _align_feature_axis(selector, tensor.shape[1])
    shape = []
    for i in range(tensor.ndim):
        if i == 0:
            shape.append(selector.shape[0])
        elif i == 1:
            shape.append(selector.shape[1])
        elif i == dim:
            shape.append(selector.shape[2])
        else:
            shape.append(1)
    return selector.reshape(shape)


def _index_for_dim(index_2d: Tensor, tensor: Tensor, dim: int) -> Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    index_2d = _align_feature_axis(index_2d, tensor.shape[1]).to(dtype=torch.long)
    view_shape = [index_2d.shape[0], index_2d.shape[1]] + [1] * (tensor.ndim - 2)
    index = index_2d.view(*view_shape)
    expand_shape = list(tensor.shape)
    expand_shape[dim] = 1
    return index.expand(*expand_shape)


def ensure_diff_ctx(
    sampling_ctx: SamplingContext | None,
    *,
    batch_size: int,
    features: int,
    device: torch.device,
    method: str | DiffSampleMethod,
    tau: float,
    hard: bool,
) -> DifferentiableSamplingContext:
    """Normalize a context to DifferentiableSamplingContext with routing metadata."""
    method_enum = _as_method(method)
    if isinstance(sampling_ctx, DifferentiableSamplingContext):
        sampling_ctx.method = method_enum
        sampling_ctx.tau = tau
        sampling_ctx.hard = hard
        return sampling_ctx

    repetition_index = getattr(sampling_ctx, "repetition_idx", None) if sampling_ctx is not None else None
    return DifferentiableSamplingContext(
        channel_index=torch.zeros((batch_size, features), dtype=torch.long, device=device),
        mask=torch.ones((batch_size, features), dtype=torch.bool, device=device),
        repetition_index=repetition_index,
        method=method_enum,
        tau=tau,
        hard=hard,
    )


def select_parent_axis(tensor: Tensor, *, sampling_ctx: SamplingContext, dim: int) -> Tensor:
    """Select a parent-routed axis using soft selectors when available."""
    channel_select = getattr(sampling_ctx, "channel_select", None)
    if channel_select is not None:
        selector = _selector_for_dim(channel_select, tensor, dim)
        return select_with_soft_or_hard(tensor, selector=selector, dim=dim)
    index = _index_for_dim(sampling_ctx.channel_index, tensor, dim)
    return tensor.gather(dim=dim, index=index).squeeze(dim)


def select_repetition_axis(
    tensor: Tensor,
    *,
    sampling_ctx: SamplingContext,
    dim: int,
    num_repetitions: int,
    strict_diff_requires_selector: bool = False,
) -> Tensor:
    """Select repetition-routed axis using repetition selectors or indices."""
    repetition_select = getattr(sampling_ctx, "repetition_select", None)
    if repetition_select is not None:
        selector = _selector_for_dim(repetition_select, tensor, dim)
        return select_with_soft_or_hard(tensor, selector=selector, dim=dim)

    if strict_diff_requires_selector and isinstance(sampling_ctx, DifferentiableSamplingContext):
        if getattr(sampling_ctx, "is_differentiable", False) and num_repetitions > 1:
            raise ValueError(
                "DifferentiableSamplingContext requires repetition_select for differentiable repetition routing."
            )

    repetition_idx = sampling_ctx.repetition_idx
    if repetition_idx is None:
        if num_repetitions > 1:
            raise ValueError("repetition_idx must be provided when sampling with num_repetitions > 1")
        if tensor.shape[dim] == 1:
            return tensor.squeeze(dim)
        return tensor.narrow(dim, 0, 1).squeeze(dim)

    if repetition_idx.ndim == 1:
        repetition_idx = repetition_idx.unsqueeze(1)
    index = _index_for_dim(repetition_idx, tensor, dim)
    return tensor.gather(dim=dim, index=index).squeeze(dim)


def condition_logits_with_evidence(logits: Tensor, evidence_ll: Tensor | None, *, dim: int) -> Tensor:
    """Condition logits on evidence by computing posterior log-probabilities."""
    if evidence_ll is None:
        return logits
    return (logits + evidence_ll).log_softmax(dim=dim)


def sample_selector_and_index(
    *,
    logits: Tensor,
    dim: int,
    is_mpe: bool,
    method: str | DiffSampleMethod,
    tau: float,
    hard: bool,
) -> tuple[Tensor, Tensor]:
    """Sample differentiable selector and corresponding hard index."""
    selector = sample_categorical_differentiably(
        dim=dim,
        is_mpe=is_mpe,
        hard=hard,
        tau=tau,
        logits=logits,
        method=_as_method(method),
    )
    indices = selector.argmax(dim=dim)
    return selector, indices


def update_ctx_channel_routing(
    sampling_ctx: SamplingContext,
    *,
    channel_index: Tensor,
    channel_select: Tensor,
    method: str | DiffSampleMethod,
    tau: float,
    hard: bool,
) -> None:
    """Update channel routing tensors and diff-sampling metadata in context."""
    if channel_index.shape != sampling_ctx.mask.shape:
        new_mask = _align_feature_axis(sampling_ctx.mask, channel_index.shape[1]).contiguous()
        if new_mask.shape != channel_index.shape:
            new_mask = torch.ones(channel_index.shape, dtype=torch.bool, device=channel_index.device)
        sampling_ctx.update(channel_index=channel_index, mask=new_mask)
    else:
        sampling_ctx.channel_index = channel_index

    sampling_ctx.channel_select = channel_select
    if isinstance(sampling_ctx, DifferentiableSamplingContext):
        sampling_ctx.method = _as_method(method)
        sampling_ctx.tau = tau
        sampling_ctx.hard = hard


def update_ctx_repetition_routing(
    sampling_ctx: SamplingContext,
    *,
    repetition_index: Tensor,
    repetition_select: Tensor,
    method: str | DiffSampleMethod,
    tau: float,
    hard: bool,
) -> None:
    """Update repetition routing tensors and diff-sampling metadata in context."""
    sampling_ctx.repetition_idx = repetition_index
    sampling_ctx.repetition_select = repetition_select
    if isinstance(sampling_ctx, DifferentiableSamplingContext):
        sampling_ctx.method = _as_method(method)
        sampling_ctx.tau = tau
        sampling_ctx.hard = hard


def merge_disjoint_child_outputs(base_data: Tensor, child_outputs: Sequence[Tensor]) -> Tensor:
    """Merge disjoint-scope child samples into one tensor."""
    result = base_data.clone()
    for child in child_outputs:
        fill_mask = torch.isnan(result) & torch.isfinite(child)
        result[fill_mask] = child[fill_mask]
    return result


def blend_same_scope_child_outputs(
    base_data: Tensor,
    *,
    child_outputs: Sequence[Tensor],
    branch_weights: Tensor,
    scope_cols: Tensor,
) -> Tensor:
    """Blend same-scope child samples using differentiable branch weights."""
    result = base_data.clone()
    if scope_cols.numel() == 0:
        return result

    stacked = torch.stack([child[:, scope_cols] for child in child_outputs], dim=-1)
    weights = _align_feature_axis(branch_weights, stacked.shape[1])
    blended = (stacked * weights).sum(dim=-1)
    result[:, scope_cols] = blended
    return result
