"""Sampling context utilities for SPFlow probabilistic circuits.

This module provides context management for both hard and differentiable
sampling operations in SPFlow. Sampling contexts track routing state across
recursive module calls.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError, ShapeError, UnsupportedOperationError

DiffSampleMethod = Literal["simple", "gumbel"]


def _resolve_device(device: torch.device | None) -> torch.device:
    """Resolve target tensor device for context allocation."""
    if device is not None:
        return device
    if hasattr(torch, "get_default_device"):
        return torch.get_default_device()
    return torch.device("cpu")


def _check_mask_bool(mask: Tensor) -> None:
    """Check if mask tensor has boolean dtype."""
    if not mask.dtype == torch.bool:
        raise InvalidParameterError("Mask must be of type torch.bool.")


def _raise_feature_width_mismatch(*, got: int, expected: str) -> None:
    """Raise a consistent feature-width mismatch error."""
    raise ShapeError(
        "Received incompatible sampling context feature width: " f"got {got}, expected {expected}."
    )


class AbstractSamplingContext(ABC):
    """Abstract base class for sampling contexts.

    Shared state:
    - `mask` indicating which batch/feature positions are active.
    - `device` where context tensors are stored.
    - `num_samples` derived from context tensors.
    """

    def __init__(
        self,
        *,
        num_samples: int | None = None,
        device: torch.device | None = None,
        mask: Tensor | None = None,
    ) -> None:
        if mask is None:
            if num_samples is None:
                raise InvalidParameterError("num_samples must be provided when mask is None.")
            resolved_device = _resolve_device(device)
            mask = torch.full((num_samples, 1), True, dtype=torch.bool, device=resolved_device)
        else:
            _check_mask_bool(mask)
            if num_samples is not None and num_samples != mask.shape[0]:
                raise InvalidParameterError(
                    "num_samples must be equal to the number of samples in mask or be ommitted."
                )

        self._mask = mask
        self.device = mask.device

    @property
    def mask(self) -> Tensor:
        """Boolean routing mask with shape `(num_samples, num_features)`."""
        return self._mask

    @mask.setter
    def mask(self, mask: Tensor) -> None:
        _check_mask_bool(mask)
        if mask.shape[0] != self.num_samples:
            raise InvalidParameterError(
                "New mask has incompatible batch dimension: "
                f"got {mask.shape[0]}, expected {self.num_samples}."
            )
        self._mask = mask

    @property
    def num_samples(self) -> int:
        """Number of samples represented by this context."""
        return int(self.mask.shape[0])

    @property
    def samples_mask(self) -> Tensor:
        """Row-level active-sample mask."""
        return self.mask.sum(1) > 0

    def require_feature_width(self, expected_features: int) -> None:
        """Assert exact feature width on this sampling context."""
        got_features = int(self._feature_width())
        if got_features != expected_features:
            _raise_feature_width_mismatch(got=got_features, expected=str(expected_features))

    @abstractmethod
    def _feature_width(self) -> int:
        """Return context feature width used for routing."""

    @abstractmethod
    def copy(self) -> AbstractSamplingContext:
        """Return a deep copy of this sampling context."""


class SamplingContext(AbstractSamplingContext):
    """Hard-routing sampling context using per-feature channel indices."""

    def __init__(
        self,
        num_samples: int | None = None,
        device: torch.device | None = None,
        channel_index: Tensor | None = None,
        mask: Tensor | None = None,
        repetition_index: Tensor | None = None,
    ) -> None:
        if channel_index is not None and mask is not None:
            if not channel_index.shape == mask.shape:
                raise InvalidParameterError("channel_index and mask must have the same shape.")

            if num_samples is not None and num_samples != channel_index.shape[0]:
                raise InvalidParameterError(
                    "num_samples must be equal to the number of samples in channel_index or be ommitted."
                )

        if channel_index is not None and mask is None:
            if num_samples is not None and num_samples != channel_index.shape[0]:
                raise InvalidParameterError(
                    "num_samples must be equal to the number of samples in channel_index or be ommitted."
                )
            num_samples = channel_index.shape[0]

        if channel_index is None and mask is not None:
            if num_samples is not None and num_samples != mask.shape[0]:
                raise InvalidParameterError(
                    "num_samples must be equal to the number of samples in mask or be ommitted."
                )
            num_samples = mask.shape[0]

        if (channel_index is None) ^ (mask is None):
            raise InvalidParameterError("channel_index and mask must be both None or both not None.")

        if channel_index is None and mask is None:
            if num_samples is None:
                raise InvalidParameterError(
                    "num_samples must be provided when channel_index and mask are None."
                )
            resolved_device = _resolve_device(device)
            channel_index = torch.zeros((num_samples, 1), dtype=torch.long, device=resolved_device)
            mask = torch.full((num_samples, 1), True, dtype=torch.bool, device=resolved_device)

        super().__init__(num_samples=num_samples, device=device, mask=mask)

        if channel_index is None:
            raise InvalidParameterError("channel_index could not be initialized.")
        if channel_index.shape != self.mask.shape:
            raise InvalidParameterError("channel_index and mask must have the same shape.")

        self._channel_index = channel_index
        if repetition_index is None:
            repetition_index = torch.zeros((self.num_samples,), dtype=torch.long, device=self.device)
        self.repetition_idx = repetition_index

    def update(self, channel_index: Tensor, mask: Tensor) -> None:
        """Update the hard-routing state (`channel_index` and `mask`)."""
        if not channel_index.shape == mask.shape:
            raise InvalidParameterError("channel_index and mask must have the same shape.")

        _check_mask_bool(mask)

        self._channel_index = channel_index
        self._mask = mask

    @property
    def channel_index(self) -> Tensor:
        return self._channel_index

    @channel_index.setter
    def channel_index(self, channel_index: Tensor) -> None:
        if channel_index.shape != self._mask.shape:
            raise InvalidParameterError("New channel_index and previous mask must have the same shape.")
        self._channel_index = channel_index

    @property
    def mask(self) -> Tensor:
        return self._mask

    @mask.setter
    def mask(self, mask: Tensor) -> None:
        if mask.shape != self._channel_index.shape:
            raise InvalidParameterError("New mask and previous channel_index must have the same shape.")
        _check_mask_bool(mask)
        self._mask = mask

    @property
    def channel_index_masked(self) -> Tensor:
        return self.channel_index[self.samples_mask]

    def copy(self) -> SamplingContext:
        """Return a copy of the sampling context."""
        return SamplingContext(
            channel_index=self.channel_index.clone(),
            mask=self.mask.clone(),
            repetition_index=self.repetition_idx.clone() if self.repetition_idx is not None else None,
        )

    def _feature_width(self) -> int:
        return int(self.channel_index.shape[1])

    def broadcast_feature_width(self, target_features: int, allow_from_one: bool = True) -> None:
        """Expand singleton feature routing to a target feature width."""
        current_features = int(self.channel_index.shape[1])
        if current_features == target_features:
            return

        if allow_from_one and current_features == 1:
            self.update(
                channel_index=self.channel_index.repeat(1, target_features),
                mask=self.mask.repeat(1, target_features),
            )
            return

        expected = f"{target_features} or 1" if allow_from_one else str(target_features)
        _raise_feature_width_mismatch(got=current_features, expected=expected)

    def repeat_split_feature_width(self, num_splits: int, target_features: int) -> None:
        """Expand split-sized contexts to full input width by feature repetition."""
        if num_splits < 1:
            raise InvalidParameterError(f"num_splits must be >= 1, got {num_splits}.")

        current_features = int(self.channel_index.shape[1])
        if current_features == target_features:
            return

        if target_features % num_splits != 0:
            raise ShapeError(
                "Cannot adapt split-sized sampling context: "
                f"target features {target_features} are not divisible by num_splits {num_splits}."
            )

        split_features = target_features // num_splits
        if current_features != split_features:
            _raise_feature_width_mismatch(
                got=current_features,
                expected=f"{target_features} or split width {split_features}",
            )

        self.update(
            channel_index=self.channel_index.repeat(1, num_splits),
            mask=self.mask.repeat(1, num_splits),
        )

    def scatter_split_groups_to_input_width(
        self,
        index_groups: Sequence[Sequence[int]],
        input_features: int,
    ) -> None:
        """Scatter split-sized routing tensors into full input-feature positions."""
        current_features = int(self.channel_index.shape[1])
        if current_features == input_features:
            return

        flat_indices: list[int] = [idx for group in index_groups for idx in group]
        if sorted(flat_indices) != list(range(input_features)):
            raise InvalidParameterError("index_groups must cover all input features exactly once.")

        split_sizes = [len(group) for group in index_groups]
        if any(size != current_features for size in split_sizes):
            _raise_feature_width_mismatch(
                got=current_features,
                expected=f"{input_features} or common split width {split_sizes}",
            )

        channel_index = self.channel_index.new_zeros((self.channel_index.shape[0], input_features))
        mask = self.mask.new_zeros((self.mask.shape[0], input_features))
        for group in index_groups:
            dest = torch.as_tensor(group, dtype=torch.long, device=self.channel_index.device)
            channel_index[:, dest] = self.channel_index
            mask[:, dest] = self.mask
        self.update(channel_index=channel_index, mask=mask)

    def slice_feature_ranges(self, ranges: Sequence[tuple[int, int]]) -> list[tuple[Tensor, Tensor]]:
        """Slice per-child contiguous feature ranges from a full-width context."""
        ctx_features = int(self.channel_index.shape[1])
        out: list[tuple[Tensor, Tensor]] = []
        for start, end in ranges:
            if start < 0 or end < start or end > ctx_features:
                raise InvalidParameterError(
                    f"Received invalid feature slice range ({start}, {end}) "
                    f"for context width {ctx_features}."
                )
            out.append((self.channel_index[:, start:end], self.mask[:, start:end]))
        return out

    def route_channel_offsets(
        self,
        child_channel_counts: Sequence[int],
    ) -> list[tuple[Tensor, Tensor]]:
        """Route global channel ids into child-local channel ids with masks."""
        if any(count < 1 for count in child_channel_counts):
            raise InvalidParameterError("child_channel_counts must all be >= 1.")

        total_channels = int(sum(child_channel_counts))
        out: list[tuple[Tensor, Tensor]] = []
        global_channel_index = self.channel_index
        active_channels = global_channel_index[self.mask]
        if active_channels.numel() > 0:
            invalid = (active_channels < 0) | (active_channels >= total_channels)
            if invalid.any():
                invalid_values = active_channels[invalid]
                observed_min = int(invalid_values.min().item())
                observed_max = int(invalid_values.max().item())
                raise InvalidParameterError(
                    "sampling_ctx.channel_index contains out-of-range channel ids on active positions: "
                    f"valid range is [0, {total_channels - 1}], observed min={observed_min}, "
                    f"max={observed_max}."
                )

        offset = 0
        for child_channels in child_channel_counts:
            child_start = offset
            child_end = offset + child_channels
            in_child_range = (global_channel_index >= child_start) & (global_channel_index < child_end)
            local_channel_index = global_channel_index - child_start
            local_channel_index = torch.where(
                in_child_range,
                local_channel_index,
                torch.zeros_like(local_channel_index),
            )
            child_mask = in_child_range & self.mask
            out.append((local_channel_index, child_mask))
            offset = child_end
        return out

    def __repr__(self) -> str:
        return (
            "SamplingContext("
            f"channel_index.shape={tuple(self.channel_index.shape)}, "
            f"mask.shape={tuple(self.mask.shape)}, "
            f"num_samples={self.num_samples}"
            ")"
        )


class DifferentiableSamplingContext(AbstractSamplingContext):
    """Differentiable sampling context using probability routing tensors."""

    def __init__(
        self,
        num_samples: int | None = None,
        device: torch.device | None = None,
        channel_probs: Tensor | None = None,
        mask: Tensor | None = None,
        repetition_probs: Tensor | None = None,
        diff_method: DiffSampleMethod = "simple",
        hard: bool = False,
        temperature_sums: float = 1.0,
        temperature_leaves: float = 1.0,
        sample_accum: Tensor | None = None,
        sample_mass: Tensor | None = None,
    ) -> None:
        if diff_method not in {"simple", "gumbel"}:
            raise InvalidParameterError(
                f"diff_method must be one of ('simple', 'gumbel'), got {diff_method!r}."
            )
        if temperature_sums <= 0.0:
            raise InvalidParameterError(f"temperature_sums must be > 0, got {temperature_sums}.")
        if temperature_leaves <= 0.0:
            raise InvalidParameterError(f"temperature_leaves must be > 0, got {temperature_leaves}.")

        if (channel_probs is None) ^ (mask is None):
            raise InvalidParameterError("channel_probs and mask must be both None or both not None.")

        if channel_probs is None and mask is None:
            if num_samples is None:
                raise InvalidParameterError(
                    "num_samples must be provided when channel_probs and mask are None."
                )
            resolved_device = _resolve_device(device)
            mask = torch.full((num_samples, 1), True, dtype=torch.bool, device=resolved_device)
            channel_probs = torch.ones(
                (num_samples, 1, 1),
                dtype=torch.get_default_dtype(),
                device=resolved_device,
            )
        else:
            if channel_probs is None or mask is None:
                raise InvalidParameterError("channel_probs and mask must be both None or both not None.")
            if channel_probs.ndim != 3:
                raise InvalidParameterError(
                    "channel_probs must have shape (num_samples, num_features, num_channels)."
                )
            if channel_probs.shape[0] != mask.shape[0] or channel_probs.shape[1] != mask.shape[1]:
                raise InvalidParameterError("channel_probs and mask must match on batch and feature axes.")
            if num_samples is not None and num_samples != channel_probs.shape[0]:
                raise InvalidParameterError(
                    "num_samples must be equal to the number of samples in channel_probs or be ommitted."
                )

        super().__init__(num_samples=num_samples, device=device, mask=mask)

        if channel_probs is None:
            raise InvalidParameterError("channel_probs could not be initialized.")
        self._validate_channel_probs(channel_probs, self.mask)
        self._channel_probs = channel_probs

        if repetition_probs is None:
            repetition_probs = torch.ones(
                (self.num_samples, 1),
                dtype=self.channel_probs.dtype,
                device=self.channel_probs.device,
            )
        else:
            self._validate_repetition_probs(repetition_probs, self.num_samples)
        self.repetition_probs = repetition_probs

        self.diff_method: DiffSampleMethod = diff_method
        self.hard = bool(hard)
        self.temperature_sums = float(temperature_sums)
        self.temperature_leaves = float(temperature_leaves)

        self.sample_accum = sample_accum.clone() if sample_accum is not None else None
        self.sample_mass = sample_mass.clone() if sample_mass is not None else None

        if self.sample_accum is not None and self.sample_accum.shape[0] != self.num_samples:
            raise InvalidParameterError(
                "sample_accum has incompatible batch size: "
                f"got {self.sample_accum.shape[0]}, expected {self.num_samples}."
            )
        if self.sample_mass is not None and self.sample_mass.shape[0] != self.num_samples:
            raise InvalidParameterError(
                "sample_mass has incompatible batch size: "
                f"got {self.sample_mass.shape[0]}, expected {self.num_samples}."
            )
        if self.sample_accum is not None and self.sample_mass is not None:
            if self.sample_accum.shape != self.sample_mass.shape:
                raise InvalidParameterError(
                    "sample_accum and sample_mass must have the same shape, "
                    f"got {tuple(self.sample_accum.shape)} and {tuple(self.sample_mass.shape)}."
                )

    @staticmethod
    def _validate_channel_probs(channel_probs: Tensor, mask: Tensor) -> None:
        if channel_probs.ndim != 3:
            raise InvalidParameterError(
                "channel_probs must have shape (num_samples, num_features, num_channels)."
            )
        if channel_probs.shape[2] < 1:
            raise InvalidParameterError("channel_probs must have at least one channel.")

        active_probs = channel_probs[mask]
        if active_probs.numel() == 0:
            return

        if (active_probs < 0).any():
            raise InvalidParameterError("channel_probs must be non-negative on active positions.")

        simplex_mass = active_probs.sum(dim=-1)
        if not torch.allclose(
            simplex_mass,
            torch.ones_like(simplex_mass),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise InvalidParameterError(
                "channel_probs must sum to 1 on active positions (within atol=1e-4, rtol=1e-4)."
            )

    def _set_channel_probs_and_mask(self, channel_probs: Tensor, mask: Tensor) -> None:
        """Set `channel_probs` and `mask` together with joint validation."""
        _check_mask_bool(mask)
        if channel_probs.shape[0] != mask.shape[0] or channel_probs.shape[1] != mask.shape[1]:
            raise InvalidParameterError("channel_probs and mask must match on batch and feature axes.")
        self._validate_channel_probs(channel_probs, mask)
        self._channel_probs = channel_probs
        self._mask = mask

    @staticmethod
    def _validate_repetition_probs(repetition_probs: Tensor, num_samples: int) -> None:
        if repetition_probs.ndim != 2:
            raise InvalidParameterError("repetition_probs must have shape (num_samples, num_repetitions).")
        if repetition_probs.shape[0] != num_samples:
            raise InvalidParameterError(
                "repetition_probs has incompatible batch size: "
                f"got {repetition_probs.shape[0]}, expected {num_samples}."
            )
        if repetition_probs.shape[1] < 1:
            raise InvalidParameterError("repetition_probs must have at least one repetition.")

        if (repetition_probs < 0).any():
            raise InvalidParameterError("repetition_probs must be non-negative.")

        simplex_mass = repetition_probs.sum(dim=-1)
        if not torch.allclose(
            simplex_mass,
            torch.ones_like(simplex_mass),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise InvalidParameterError(
                "repetition_probs must sum to 1 per sample (within atol=1e-4, rtol=1e-4)."
            )

    @property
    def channel_probs(self) -> Tensor:
        return self._channel_probs

    @channel_probs.setter
    def channel_probs(self, channel_probs: Tensor) -> None:
        self._set_channel_probs_and_mask(channel_probs, self.mask)

    @property
    def mask(self) -> Tensor:
        return self._mask

    @mask.setter
    def mask(self, mask: Tensor) -> None:
        if hasattr(self, "_channel_probs"):
            self._set_channel_probs_and_mask(self._channel_probs, mask)
            return
        _check_mask_bool(mask)
        self._mask = mask

    @property
    def channel_index(self) -> Tensor:
        raise UnsupportedOperationError(
            "DifferentiableSamplingContext does not support channel_index; use channel_probs instead."
        )

    @channel_index.setter
    def channel_index(self, channel_index: Tensor) -> None:
        del channel_index
        raise UnsupportedOperationError(
            "DifferentiableSamplingContext does not support channel_index; use channel_probs instead."
        )

    @property
    def repetition_idx(self) -> Tensor:
        raise UnsupportedOperationError(
            "DifferentiableSamplingContext does not support repetition_idx; use repetition_probs instead."
        )

    @repetition_idx.setter
    def repetition_idx(self, repetition_idx: Tensor) -> None:
        del repetition_idx
        raise UnsupportedOperationError(
            "DifferentiableSamplingContext does not support repetition_idx; use repetition_probs instead."
        )

    @property
    def repetition_index(self) -> Tensor:
        raise UnsupportedOperationError(
            "DifferentiableSamplingContext does not support repetition_index; use repetition_probs instead."
        )

    @repetition_index.setter
    def repetition_index(self, repetition_index: Tensor) -> None:
        del repetition_index
        raise UnsupportedOperationError(
            "DifferentiableSamplingContext does not support repetition_index; use repetition_probs instead."
        )

    def update(self, channel_index: Tensor, mask: Tensor) -> None:
        del channel_index
        del mask
        raise UnsupportedOperationError(
            "DifferentiableSamplingContext does not support index-based updates; update channel_probs/mask directly."
        )

    def update_prob_routing(self, channel_probs: Tensor, mask: Tensor) -> None:
        """Atomically update probability-routing tensors."""
        self._set_channel_probs_and_mask(channel_probs, mask)

    def _feature_width(self) -> int:
        return int(self.channel_probs.shape[1])

    def copy(self) -> DifferentiableSamplingContext:
        """Return a copy of the differentiable sampling context."""
        return DifferentiableSamplingContext(
            channel_probs=self.channel_probs.clone(),
            mask=self.mask.clone(),
            repetition_probs=self.repetition_probs.clone() if self.repetition_probs is not None else None,
            diff_method=self.diff_method,
            hard=self.hard,
            temperature_sums=self.temperature_sums,
            temperature_leaves=self.temperature_leaves,
            sample_accum=self.sample_accum.clone() if self.sample_accum is not None else None,
            sample_mass=self.sample_mass.clone() if self.sample_mass is not None else None,
        )

    def slice_feature_prob_ranges(self, ranges: Sequence[tuple[int, int]]) -> list[tuple[Tensor, Tensor]]:
        """Slice per-child contiguous feature ranges from a full-width probability context."""
        ctx_features = int(self.channel_probs.shape[1])
        out: list[tuple[Tensor, Tensor]] = []
        for start, end in ranges:
            if start < 0 or end < start or end > ctx_features:
                raise InvalidParameterError(
                    f"Received invalid feature slice range ({start}, {end}) "
                    f"for context width {ctx_features}."
                )
            out.append((self.channel_probs[:, start:end, :], self.mask[:, start:end]))
        return out

    def route_channel_prob_offsets(
        self,
        child_channel_counts: Sequence[int],
    ) -> list[tuple[Tensor, Tensor]]:
        """Route global channel probabilities into child-local channel probability tensors."""
        if any(count < 1 for count in child_channel_counts):
            raise InvalidParameterError("child_channel_counts must all be >= 1.")

        total_channels = int(sum(child_channel_counts))
        got_channels = int(self.channel_probs.shape[2])
        if got_channels != total_channels:
            raise ShapeError(
                "Received incompatible channel_probs width for channel routing: "
                f"got {got_channels}, expected {total_channels}."
            )

        out: list[tuple[Tensor, Tensor]] = []
        offset = 0
        for child_channels in child_channel_counts:
            child_start = offset
            child_end = offset + child_channels
            child_probs = self.channel_probs[:, :, child_start:child_end]
            child_mask = self.mask & (child_probs.sum(dim=-1) > 0.0)
            out.append((child_probs, child_mask))
            offset = child_end
        return out

    def repeat_split_feature_prob_width(self, num_splits: int, target_features: int) -> None:
        """Expand split-sized probability routing to full input width by repetition."""
        if num_splits < 1:
            raise InvalidParameterError(f"num_splits must be >= 1, got {num_splits}.")

        current_features = int(self.channel_probs.shape[1])
        if current_features == target_features:
            return

        if target_features % num_splits != 0:
            raise ShapeError(
                "Cannot adapt split-sized sampling context: "
                f"target features {target_features} are not divisible by num_splits {num_splits}."
            )

        split_features = target_features // num_splits
        if current_features != split_features:
            _raise_feature_width_mismatch(
                got=current_features,
                expected=f"{target_features} or split width {split_features}",
            )

        self._set_channel_probs_and_mask(
            self.channel_probs.repeat(1, num_splits, 1),
            self.mask.repeat(1, num_splits),
        )

    def scatter_split_prob_groups_to_input_width(
        self,
        index_groups: Sequence[Sequence[int]],
        input_features: int,
    ) -> None:
        """Scatter split-sized probability routing tensors to full input-feature positions."""
        current_features = int(self.channel_probs.shape[1])
        if current_features == input_features:
            return

        flat_indices: list[int] = [idx for group in index_groups for idx in group]
        if sorted(flat_indices) != list(range(input_features)):
            raise InvalidParameterError("index_groups must cover all input features exactly once.")

        split_sizes = [len(group) for group in index_groups]
        if any(size != current_features for size in split_sizes):
            _raise_feature_width_mismatch(
                got=current_features,
                expected=f"{input_features} or common split width {split_sizes}",
            )

        channel_probs = self.channel_probs.new_zeros(
            (self.channel_probs.shape[0], input_features, self.channel_probs.shape[2])
        )
        mask = self.mask.new_zeros((self.mask.shape[0], input_features))
        for group in index_groups:
            dest = torch.as_tensor(group, dtype=torch.long, device=self.channel_probs.device)
            channel_probs[:, dest, :] = self.channel_probs
            mask[:, dest] = self.mask
        self._set_channel_probs_and_mask(channel_probs, mask)

    def broadcast_feature_prob_width(self, target_features: int, allow_from_one: bool = True) -> None:
        """Expand singleton feature probabilities to a target feature width."""
        current_features = int(self.channel_probs.shape[1])
        if current_features == target_features:
            return

        if allow_from_one and current_features == 1:
            self._set_channel_probs_and_mask(
                self.channel_probs.repeat(1, target_features, 1),
                self.mask.repeat(1, target_features),
            )
            return

        expected = f"{target_features} or 1" if allow_from_one else str(target_features)
        _raise_feature_width_mismatch(got=current_features, expected=expected)

    def resolve_repetition_probs(
        self,
        *,
        expected_repetitions: int,
        module_name: str,
    ) -> Tensor:
        """Resolve repetition probabilities for differentiable routing.

        Args:
            expected_repetitions: Expected repetition width for the current module.
            module_name: Module identifier used in error messages.
        """
        if expected_repetitions < 1:
            raise InvalidParameterError(f"expected_repetitions must be >= 1, got {expected_repetitions}.")

        if self.repetition_probs is None:
            if expected_repetitions > 1:
                raise InvalidParameterError(
                    "sampling_ctx.repetition_probs must be provided when differentiably sampling "
                    f"{module_name} with num_repetitions > 1."
                )
            return torch.ones(
                (self.num_samples, 1),
                dtype=self.channel_probs.dtype,
                device=self.channel_probs.device,
            )

        repetition_probs = self.repetition_probs
        if repetition_probs.shape[1] == expected_repetitions:
            return repetition_probs

        if repetition_probs.shape[1] == 1 and expected_repetitions > 1:
            # Root differentiable contexts start with singleton repetition routing.
            # Align with hard sampling semantics by routing all mass to repetition 0.
            expanded = repetition_probs.new_zeros((self.num_samples, expected_repetitions))
            expanded[:, 0] = repetition_probs[:, 0]
            return expanded

        raise ShapeError(
            "sampling_ctx.repetition_probs has incompatible repetition width for "
            f"{module_name}: got {repetition_probs.shape[1]}, "
            f"expected {expected_repetitions}."
        )

    def resolve_channel_probs(
        self,
        *,
        expected_channels: int,
        module_name: str,
    ) -> Tensor:
        """Resolve channel routing probabilities for a module channel width.

        Root differentiable contexts are initialized with a singleton channel axis.
        When a module expects more channels, we map that singleton mass to channel 0,
        mirroring hard-sampling behavior where root channel indices initialize to 0.
        """
        if expected_channels < 1:
            raise InvalidParameterError(f"expected_channels must be >= 1, got {expected_channels}.")

        channel_probs = self.channel_probs
        got_channels = int(channel_probs.shape[2])
        if got_channels == expected_channels:
            return channel_probs

        if got_channels == 1 and expected_channels > 1:
            expanded = channel_probs.new_zeros(
                (channel_probs.shape[0], channel_probs.shape[1], expected_channels)
            )
            expanded[:, :, 0] = channel_probs[:, :, 0]
            return expanded

        raise ShapeError(
            "sampling_ctx.channel_probs has incompatible channel width for "
            f"{module_name}: got {got_channels}, expected {expected_channels}."
        )

    def initialize_accumulators(
        self,
        *,
        num_features: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize additive sampling accumulators."""
        if num_features < 1:
            raise InvalidParameterError(f"num_features must be >= 1, got {num_features}.")

        dtype = dtype if dtype is not None else self.channel_probs.dtype
        device = device if device is not None else self.channel_probs.device
        self.sample_accum = torch.zeros((self.num_samples, num_features), dtype=dtype, device=device)
        self.sample_mass = torch.zeros((self.num_samples, num_features), dtype=dtype, device=device)

    def accumulate_additive(
        self,
        values: Tensor,
        *,
        weights: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> None:
        """Accumulate additive differentiable sampling contributions."""
        if values.ndim != 2:
            raise ShapeError(
                "values for additive accumulation must be 2D with shape " "(num_samples, num_features)."
            )

        if values.shape[0] != self.num_samples:
            raise ShapeError(
                "values for additive accumulation have incompatible batch size: "
                f"got {values.shape[0]}, expected {self.num_samples}."
            )

        if self.sample_accum is None or self.sample_mass is None:
            self.initialize_accumulators(
                num_features=int(values.shape[1]), dtype=values.dtype, device=values.device
            )

        if self.sample_accum is None or self.sample_mass is None:
            raise InvalidParameterError("Failed to initialize additive sampling accumulators.")

        if values.shape != self.sample_accum.shape:
            raise ShapeError(
                "values for additive accumulation have incompatible shape: "
                f"got {tuple(values.shape)}, expected {tuple(self.sample_accum.shape)}."
            )

        if weights is None:
            weights = torch.ones_like(values)
        if weights.shape != values.shape:
            raise ShapeError(
                "weights for additive accumulation must match values shape, "
                f"got {tuple(weights.shape)} and {tuple(values.shape)}."
            )

        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        _check_mask_bool(mask)
        if mask.shape != values.shape:
            raise ShapeError(
                "mask for additive accumulation must match values shape, "
                f"got {tuple(mask.shape)} and {tuple(values.shape)}."
            )

        weighted_values = values * weights
        weighted_values = torch.where(mask, weighted_values, torch.zeros_like(weighted_values))
        masked_weights = torch.where(mask, weights, torch.zeros_like(weights))

        self.sample_accum = self.sample_accum + weighted_values
        self.sample_mass = self.sample_mass + masked_weights

    def finalize_with_evidence(self, data: Tensor) -> Tensor:
        """Finalize additive sampling output by filling only NaN positions in evidence."""
        if self.sample_accum is None or self.sample_mass is None:
            raise InvalidParameterError(
                "Cannot finalize differentiable sampling output before accumulators are initialized."
            )

        if data.shape != self.sample_accum.shape:
            raise ShapeError(
                "data has incompatible shape for finalize_with_evidence: "
                f"got {tuple(data.shape)}, expected {tuple(self.sample_accum.shape)}."
            )

        out = data.clone()
        fill_mask = torch.isnan(out)
        out[fill_mask] = self.sample_accum[fill_mask]
        return out

    def __repr__(self) -> str:
        return (
            "DifferentiableSamplingContext("
            f"channel_probs.shape={tuple(self.channel_probs.shape)}, "
            f"mask.shape={tuple(self.mask.shape)}, "
            f"num_samples={self.num_samples}, "
            f"diff_method={self.diff_method!r}, "
            f"hard={self.hard}"
            ")"
        )


def init_default_sampling_context(
    sampling_ctx: SamplingContext | None,
    num_samples: int | None = None,
    device: torch.device | None = None,
) -> SamplingContext:
    """Initialize hard sampling context if not already initialized."""
    if sampling_ctx is not None:
        return sampling_ctx
    return SamplingContext(num_samples=num_samples, device=device)


def build_root_sampling_context(
    sampling_ctx: SamplingContext | None,
    *,
    module_name: str,
    num_samples: int,
    num_features: int,
    device: torch.device | None = None,
) -> SamplingContext:
    """Build or validate hard sampling context at a root sampling entrypoint."""
    if sampling_ctx is None:
        channel_index = torch.zeros((num_samples, num_features), dtype=torch.long, device=device)
        mask = torch.ones((num_samples, num_features), dtype=torch.bool, device=device)
        return SamplingContext(channel_index=channel_index, mask=mask)

    if sampling_ctx.channel_index.shape[0] != num_samples:
        raise InvalidParameterError(
            f"{module_name}.sample received sampling_ctx with batch={sampling_ctx.channel_index.shape[0]}, "
            f"expected {num_samples}."
        )
    if sampling_ctx.channel_index.shape != sampling_ctx.mask.shape:
        raise InvalidParameterError(
            f"{module_name}.sample received sampling_ctx with mismatched channel_index/mask shapes: "
            f"{tuple(sampling_ctx.channel_index.shape)} vs {tuple(sampling_ctx.mask.shape)}."
        )
    if sampling_ctx.channel_index.shape[1] != num_features:
        raise InvalidParameterError(
            f"{module_name}.sample received sampling_ctx with features={sampling_ctx.channel_index.shape[1]}, "
            f"expected {num_features}."
        )

    return sampling_ctx


def update_channel_index_strict(sampling_ctx: SamplingContext, new_channel_index: Tensor) -> None:
    """Replace `channel_index` while preserving context feature-layout invariants."""
    if new_channel_index.shape[0] != sampling_ctx.mask.shape[0]:
        raise ShapeError(
            "sampling_ctx.channel_index batch mismatch for update: "
            f"got {new_channel_index.shape[0]}, expected {sampling_ctx.mask.shape[0]}."
        )
    if new_channel_index.shape[1] != sampling_ctx.mask.shape[1]:
        raise ShapeError(
            "sampling_ctx.mask has incompatible feature width for sampling update: "
            f"got {sampling_ctx.mask.shape[1]}, expected {new_channel_index.shape[1]}."
        )
    sampling_ctx.channel_index = new_channel_index
