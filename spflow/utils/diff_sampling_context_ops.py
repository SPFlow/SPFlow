from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import torch
from torch import Tensor

from spflow.utils.sampling_context import SamplingContext

_DEFAULT_SELECTOR_ATTRS = ("channel_select", "repetition_select")


def _iter_feature_selectors(
    sampling_ctx: SamplingContext, attrs: Iterable[str]
) -> Iterator[tuple[str, Tensor]]:
    for attr in attrs:
        selector = getattr(sampling_ctx, attr, None)
        if selector is None or selector.dim() < 2:
            continue
        yield attr, selector


def expand_singleton_feature_selectors_inplace(
    sampling_ctx: SamplingContext,
    *,
    target_features: int,
    attrs: Iterable[str] = _DEFAULT_SELECTOR_ATTRS,
) -> None:
    """Expand selector feature axis when it is a singleton."""
    for attr, selector in _iter_feature_selectors(sampling_ctx, attrs):
        if selector.shape[1] != 1:
            continue
        expand_shape = (-1, target_features) + (-1,) * (selector.ndim - 2)
        setattr(sampling_ctx, attr, selector.expand(*expand_shape).contiguous())


def repeat_or_expand_feature_selectors_inplace(
    sampling_ctx: SamplingContext,
    *,
    source_features: int,
    target_features: int,
    repeats: int,
    attrs: Iterable[str] = _DEFAULT_SELECTOR_ATTRS,
) -> None:
    """Repeat source feature selectors, or expand singleton selectors, to target size."""
    for attr, selector in _iter_feature_selectors(sampling_ctx, attrs):
        if selector.shape[1] == target_features:
            continue

        if selector.shape[1] == 1:
            expand_shape = (-1, target_features) + (-1,) * (selector.ndim - 2)
            setattr(sampling_ctx, attr, selector.expand(*expand_shape).contiguous())
            continue

        if selector.shape[1] != source_features:
            continue

        reps = (1, repeats) + (1,) * (selector.ndim - 2)
        repeated = selector.repeat(*reps)
        if repeated.shape[1] != target_features:
            repeated = repeated[:, :target_features, ...]
        setattr(sampling_ctx, attr, repeated.contiguous())


def map_or_expand_feature_selectors_inplace(
    sampling_ctx: SamplingContext,
    *,
    mapping: Tensor,
    out_features: int,
    in_features: int,
    attrs: Iterable[str] = _DEFAULT_SELECTOR_ATTRS,
) -> None:
    """Map output-feature selectors to input-feature selectors with a feature map."""
    for attr, selector in _iter_feature_selectors(sampling_ctx, attrs):
        if selector.shape[1] == out_features:
            mapping_f = mapping.to(dtype=selector.dtype)
            mapped = torch.einsum("bio,bo...->bi...", mapping_f, selector)
            setattr(sampling_ctx, attr, mapped.contiguous())
            continue

        if selector.shape[1] == 1:
            expand_shape = (-1, in_features) + (-1,) * (selector.ndim - 2)
            setattr(sampling_ctx, attr, selector.expand(*expand_shape).contiguous())


def select_or_expand_feature_selector(
    selector: Tensor | None,
    *,
    parent_features: int,
    child_features: int,
    feature_indices: Sequence[int] | None = None,
) -> Tensor | None:
    """Return selector sliced to child feature indices or expanded from singleton."""
    if selector is None or selector.dim() < 2:
        return None

    if feature_indices is not None and selector.shape[1] == parent_features:
        return selector[:, feature_indices, ...]

    if selector.shape[1] == 1:
        expand_shape = (-1, child_features) + (-1,) * (selector.ndim - 2)
        return selector.expand(*expand_shape).contiguous()

    return None
