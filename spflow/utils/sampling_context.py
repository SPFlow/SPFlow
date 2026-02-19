"""Sampling context utilities for SPFlow probabilistic circuits.

This module provides context management for sampling operations in SPFlow,
including the SamplingContext class that tracks sampling state across
recursive module calls and utility functions for context initialization
and management.

The sampling context is essential for:
- Managing which instances and features to sample
- Handling evidence during sampling (conditional sampling)
- Tracking channel indices for multi-output modules
- Supporting repetition-based sampling structures
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError, ShapeError


def _check_mask_bool(mask: Tensor) -> None:
    """Check if mask tensor has boolean dtype.

    Args:
        mask: Tensor to check for boolean dtype.

    Raises:
        InvalidParameterError: If mask tensor does not have torch.bool dtype.
    """
    if not mask.dtype == torch.bool:
        raise InvalidParameterError("Mask must be of type torch.bool.")


def _check_integer_dtype(tensor: Tensor, *, name: str) -> None:
    """Check that a tensor has a non-boolean integer dtype."""
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise InvalidParameterError(f"{name} must have an integer dtype.")


def _check_channel_index_tensor(channel_index: Tensor, *, name: str = "channel_index") -> None:
    """Validate routing index tensor rank and dtype."""
    if channel_index.ndim != 2:
        raise InvalidParameterError(
            f"{name} must have shape (batch, features), got rank {channel_index.ndim}."
        )
    _check_integer_dtype(channel_index, name=name)


def _check_repetition_index_tensor(repetition_index: Tensor, *, name: str = "repetition_index") -> None:
    """Validate repetition-index tensor rank and dtype."""
    if repetition_index.ndim not in (1, 2):
        raise InvalidParameterError(
            f"{name} must have shape (batch,) or (batch, K), got rank {repetition_index.ndim}."
        )
    _check_integer_dtype(repetition_index, name=name)


def _normalize_repetition_index(
    repetition_index: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> Tensor:
    """Normalize repetition indices to canonical `(batch,)` long tensor."""
    _check_repetition_index_tensor(repetition_index, name=name)
    if repetition_index.ndim == 2:
        if repetition_index.shape[1] != 1:
            raise InvalidParameterError(f"{name} must have shape (batch,) or (batch, 1).")
        repetition_index = repetition_index[:, 0]
    if repetition_index.shape[0] != batch_size:
        raise InvalidParameterError(
            f"{name} batch dimension must match channel_index: "
            f"got {repetition_index.shape[0]}, expected {batch_size}."
        )
    if repetition_index.device != device:
        raise InvalidParameterError(
            f"{name} must be on the same device as channel_index: "
            f"got {repetition_index.device} and {device}."
        )
    if repetition_index.dtype != torch.long:
        repetition_index = repetition_index.to(dtype=torch.long)
    return repetition_index


def _normalize_routing_state(
    *,
    channel_index: Tensor,
    mask: Tensor,
    repetition_index: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Validate and normalize routing tensors for consistent context state."""
    _check_channel_index_tensor(channel_index, name="channel_index")
    if channel_index.dtype != torch.long:
        channel_index = channel_index.to(dtype=torch.long)

    _check_mask_bool(mask)
    if channel_index.shape != mask.shape:
        raise InvalidParameterError("channel_index and mask must have the same shape.")
    if channel_index.device != mask.device:
        raise InvalidParameterError(
            "channel_index and mask must be on the same device: "
            f"got {channel_index.device} and {mask.device}."
        )

    if repetition_index is not None:
        repetition_index = _normalize_repetition_index(
            repetition_index,
            batch_size=channel_index.shape[0],
            device=channel_index.device,
            name="repetition_index",
        )
    return channel_index, mask, repetition_index


class SamplingContext:
    """Context information manager for sampling operations in probabilistic circuits.

    Manages sampling state across recursive module calls, tracking which instances
    to sample, which output channels to use, and any evidence constraints. This
    is essential for conditional sampling, structured sampling, and maintaining
    consistency across complex circuit hierarchies. channel_index determines which output channels to use for sampling,
    mask allows selective sampling of subsets of instances/features, repetition_index supports structured circuit representations,
    and all tensors must have compatible shapes for proper operation.

    Attributes:
        num_samples (int): Number of samples to generate.
        device (torch.device): Device on which tensors are stored.
        channel_index (Tensor | None): Channel indices to sample from for each
            instance and feature. Shape: (batch_size, num_features).
        mask (Tensor | None): Boolean mask indicating which instances/features
            should be sampled. Shape matches channel_index.
        repetition_index (Tensor | None): Indices for repetition-based structures.
    """

    def __init__(
        self,
        num_samples: int | None = None,
        device: torch.device | None = None,
        channel_index: Tensor | None = None,
        mask: Tensor | None = None,
        repetition_index: Tensor | None = None,
        is_mpe: bool = False,
        is_differentiable: bool = False,
    ) -> None:
        """Initialize SamplingContext for managing sampling operations.

        Creates a sampling context with specified parameters. The context manages
        which instances and features to sample, handles evidence constraints, and supports
        structured sampling through repetition indices. Channel_index determines which output
        channels to use for sampling, mask allows selective sampling of subsets of
        instances/features, and repetition_index supports structured circuit representations.
        All tensors must have compatible shapes for proper operation.

        Args:
            num_samples (int | None, optional): Number of samples to generate.
                Can be inferred from other tensor dimensions if not provided.
                Defaults to None.
            device (torch.device | None, optional): PyTorch device for tensor storage.
                If None, uses torch.get_default_device(). Defaults to None.
            channel_index (Tensor | None, optional): Channel indices for each sample
                and feature. Shape: (batch_size, num_features). If None, samples
                from all channels. Defaults to None.
            mask (Tensor | None, optional): Boolean mask indicating which instances/
                features to sample from. Must be torch.bool type. Shape must match
                channel_index if provided. Defaults to None.
            repetition_index (Tensor | None, optional): Indices for repetition-based
                sampling structures. Used by circuits with repeated computations.
                Defaults to None.
            is_mpe (bool, optional): If True, sampling uses MPE decisions instead of
                stochastic sampling. Defaults to False.
            is_differentiable (bool, optional): If True, sampling operations will be
                differentiable. Defaults to False.

        Raises:
            InvalidParameterError: If tensor shapes are incompatible, mask has wrong dtype,
                or num_samples conflicts with tensor dimensions.
        """
        device_was_provided = device is not None
        self.is_differentiable = is_differentiable
        self.is_mpe = is_mpe
        self._channel_index: Tensor
        self._mask: Tensor
        self._repetition_index: Tensor | None
        if device is None:
            device = torch.get_default_device()

        # Allowed option 1: (num_samples)
        if num_samples is not None:
            if channel_index is not None or repetition_index is not None or mask is not None:
                raise InvalidParameterError(
                    "SamplingContext accepts either `num_samples` or (`channel_index`, `repetition_index`, "
                    "optional `mask`)."
                )
            self._assign_state(
                channel_index=torch.zeros((num_samples, 1), dtype=torch.long, device=device),
                mask=torch.full((num_samples, 1), True, dtype=torch.bool, device=device),
                repetition_index=torch.zeros((num_samples,), dtype=torch.long, device=device),
            )
            return

        # Allowed options 2/3: (channel_index[, repetition_index][, mask])
        if channel_index is None:
            raise InvalidParameterError(
                "When `num_samples` is not provided, `channel_index` must be provided."
            )

        if mask is None:
            mask = torch.ones_like(channel_index, dtype=torch.bool)
        if repetition_index is None:
            repetition_index = torch.zeros(
                (channel_index.shape[0],),
                dtype=torch.long,
                device=channel_index.device,
            )
        channel_index, mask, repetition_index = _normalize_routing_state(
            channel_index=channel_index,
            mask=mask,
            repetition_index=repetition_index,
        )
        if device_was_provided and device != channel_index.device:
            raise InvalidParameterError(
                "If `device` is provided, it must match channel_index/repetition_index/mask device: "
                f"got {device}, expected {channel_index.device}."
            )
        self._assign_state(
            channel_index=channel_index,
            mask=mask,
            repetition_index=repetition_index,
        )

    def _assign_state(
        self,
        *,
        channel_index: Tensor,
        mask: Tensor,
        repetition_index: Tensor | None,
    ) -> None:
        """Commit a fully validated routing state."""
        self._channel_index = channel_index
        self._mask = mask
        self._repetition_index = repetition_index
        self.device = channel_index.device

    @property
    def channel_index(self) -> Tensor:
        return self._channel_index

    @channel_index.setter
    def channel_index(self, channel_index: Tensor) -> None:
        _check_channel_index_tensor(channel_index, name="channel_index")
        if channel_index.dtype != torch.long:
            channel_index = channel_index.to(dtype=torch.long)
        if channel_index.device != self._mask.device:
            raise InvalidParameterError(
                "channel_index and mask must be on the same device: "
                f"got {channel_index.device} and {self._mask.device}."
            )
        if channel_index.shape[0] != self._mask.shape[0]:
            raise InvalidParameterError(
                "channel_index and mask must share the same batch size: "
                f"got {channel_index.shape[0]} and {self._mask.shape[0]}."
            )
        if self._repetition_index is not None:
            if channel_index.shape[0] != self._repetition_index.shape[0]:
                raise InvalidParameterError(
                    "channel_index batch dimension must match repetition_index: "
                    f"got {channel_index.shape[0]}, expected {self._repetition_index.shape[0]}."
                )
            if channel_index.device != self._repetition_index.device:
                raise InvalidParameterError(
                    "channel_index and repetition_index must be on the same device: "
                    f"got {channel_index.device} and {self._repetition_index.device}."
                )
        self._channel_index = channel_index
        self.device = channel_index.device

    @property
    def mask(self) -> Tensor:
        return self._mask

    @mask.setter
    def mask(self, mask: Tensor) -> None:
        _check_mask_bool(mask)
        if mask.device != self._channel_index.device:
            raise InvalidParameterError(
                "mask and channel_index must be on the same device: "
                f"got {mask.device} and {self._channel_index.device}."
            )
        if mask.shape[0] != self._channel_index.shape[0]:
            raise InvalidParameterError(
                "mask and channel_index must share the same batch size: "
                f"got {mask.shape[0]} and {self._channel_index.shape[0]}."
            )
        self._mask = mask
        self.device = mask.device

    @property
    def repetition_index(self) -> Tensor | None:
        return self._repetition_index

    @repetition_index.setter
    def repetition_index(self, repetition_index: Tensor | None) -> None:
        if repetition_index is None:
            self._repetition_index = None
            return
        self._repetition_index = _normalize_repetition_index(
            repetition_index,
            batch_size=self._channel_index.shape[0],
            device=self._channel_index.device,
            name="repetition_index",
        )

    @property
    def samples_mask(self) -> Tensor:
        return self.mask.sum(1) > 0

    @property
    def channel_index_masked(self) -> Tensor:
        return self.channel_index[self.samples_mask]

    def copy(self) -> SamplingContext:
        """Return a copy of the sampling context.

        Returns:
            SamplingContext: A new SamplingContext instance with copied tensors.
        """
        return self.with_routing(
            channel_index=self.channel_index,
            mask=self.mask,
            clone_routing=True,
            clone_repetition=True,
        )

    def with_routing(
        self,
        *,
        channel_index: Tensor,
        mask: Tensor,
        clone_routing: bool = True,
        clone_repetition: bool = True,
    ) -> SamplingContext:
        """Build a child context with routing tensors and inherited sampling flags."""
        if clone_routing:
            channel_index = channel_index.clone()
            mask = mask.clone()

        repetition_index = self.repetition_index
        if repetition_index is not None and clone_repetition:
            repetition_index = repetition_index.clone()

        return SamplingContext(
            channel_index=channel_index,
            mask=mask,
            repetition_index=repetition_index,
            is_mpe=self.is_mpe,
        )

    def require_feature_width(self, expected_features: int) -> None:
        """Assert exact feature width on this sampling context.

        Args:
            expected_features: Required feature width.

        Raises:
            ShapeError: If context width does not match `expected_features`.
        """
        got_features = int(self.channel_index.shape[1])
        if got_features != expected_features:
            raise ShapeError(
                "Received incompatible sampling context feature width: "
                f"got {got_features}, expected {expected_features}."
            )

    def broadcast_feature_width(self, target_features: int, allow_from_one: bool = True) -> None:
        """Expand singleton feature routing to a target feature width.

        The only supported implicit broadcast is from width `1` to
        `target_features`, guarded by `allow_from_one`.

        Args:
            target_features: Desired feature width.
            allow_from_one: Whether width-1 contexts may be repeated.

        Raises:
            ShapeError: If width adaptation is not allowed or impossible.
        """
        current_features = int(self.channel_index.shape[1])
        if current_features == target_features:
            return

        if allow_from_one and current_features == 1:
            # Repeat along feature axis while preserving per-row batch semantics.
            self.channel_index = self.channel_index.repeat(1, target_features)
            self.mask = self.mask.repeat(1, target_features)
            return

        if allow_from_one:
            expected = f"{target_features} or 1"
        else:
            expected = str(target_features)
        raise ShapeError(
            "Received incompatible sampling context feature width: "
            f"got {current_features}, expected {expected}."
        )

    def repeat_split_feature_width(self, num_splits: int, target_features: int) -> None:
        """Expand split-sized contexts to full input width by feature repetition.

        Args:
            num_splits: Number of split branches.
            target_features: Target full feature width.

        Raises:
            InvalidParameterError: If `num_splits < 1`.
            ShapeError: If split sizing is incompatible with the target width.
        """
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
            raise ShapeError(
                "Received incompatible sampling context feature width: "
                f"got {current_features}, expected {target_features} or split width {split_features}."
            )

        self.channel_index = self.channel_index.repeat(1, num_splits)
        self.mask = self.mask.repeat(1, num_splits)

    def scatter_split_groups_to_input_width(
        self,
        index_groups: Sequence[Sequence[int]],
        input_features: int,
    ) -> None:
        """Scatter split-sized routing tensors into full input-feature positions.

        Args:
            index_groups: Feature-index groups defining split placement.
            input_features: Required full input feature width.

        Raises:
            InvalidParameterError: If index groups are not an exact partition of
                `[0, input_features)`.
            ShapeError: If context width cannot be interpreted as a common
                split-sized width.
        """
        current_features = int(self.channel_index.shape[1])
        if current_features == input_features:
            return

        flat_indices: list[int] = [idx for group in index_groups for idx in group]
        if sorted(flat_indices) != list(range(input_features)):
            raise InvalidParameterError("index_groups must cover all input features exactly once.")

        split_sizes = [len(group) for group in index_groups]
        if any(size != current_features for size in split_sizes):
            raise ShapeError(
                "Received incompatible sampling context feature width: "
                f"got {current_features}, expected {input_features} or common split width {split_sizes}."
            )

        # Build full-width tensors and place each split routing slice at its
        # destination feature positions.
        channel_index = self.channel_index.new_zeros((self.channel_index.shape[0], input_features))
        mask = self.mask.new_zeros((self.mask.shape[0], input_features))
        for group in index_groups:
            dest = torch.as_tensor(group, dtype=torch.long, device=self.channel_index.device)
            channel_index[:, dest] = self.channel_index
            mask[:, dest] = self.mask
        self.channel_index = channel_index
        self.mask = mask

    def slice_feature_ranges(self, ranges: Sequence[tuple[int, int]]) -> list[tuple[Tensor, Tensor]]:
        """Slice per-child contiguous feature ranges from a full-width context.

        Args:
            ranges: Inclusive-exclusive `(start, end)` feature ranges.

        Returns:
            List of `(channel_index_slice, mask_slice)` tuples, one per range.

        Raises:
            InvalidParameterError: If any range is outside the context width or has
                invalid bounds.
        """
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
        """Route global channel ids into child-local channel ids with masks.

        Args:
            child_channel_counts: Number of channels per child in concatenation
                order.

        Returns:
            List of `(local_channel_index, child_mask)` tuples, one per child.

        Raises:
            InvalidParameterError: If any child channel count is less than 1, or
                if active channel ids fall outside the concatenated child range.
        """
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
                # Keep non-owned positions in-bounds for downstream gather ops;
                # correctness is enforced via child_mask and the range check above.
                torch.zeros_like(local_channel_index),
            )
            child_mask = in_child_range & self.mask
            out.append((local_channel_index, child_mask))
            offset = child_end
        return out

    def validate_sampling_context(
        self,
        *,
        num_samples: int | None = None,
        num_features: int | None = None,
        num_channels: int | None = None,
        num_repetitions: int | None = None,
        allowed_feature_widths: Sequence[int] | None = None,
    ) -> None:
        """Validate this sampling context for an internal `_sample(...)` call."""
        _check_channel_index_tensor(self.channel_index, name="sampling_ctx.channel_index")
        if self.channel_index.shape != self.mask.shape:
            raise InvalidParameterError(
                "sampling_ctx has mismatched channel_index/mask shapes: "
                f"channel_index={self.channel_index.shape}, mask={self.mask.shape}."
            )
        _check_mask_bool(self.mask)
        if self.channel_index.device != self.mask.device:
            raise InvalidParameterError(
                "sampling_ctx.channel_index and sampling_ctx.mask must be on the same device: "
                f"got {self.channel_index.device} and {self.mask.device}."
            )

        if num_samples is not None and self.channel_index.shape[0] != num_samples:
            raise InvalidParameterError(
                "sampling_ctx batch size mismatch: "
                f"got {self.channel_index.shape[0]}, "
                f"expected {num_samples}."
            )

        if allowed_feature_widths is None:
            expected_feature_widths: set[int] = set()
            if num_features is not None:
                expected_feature_widths.add(int(num_features))
        else:
            expected_feature_widths = {int(width) for width in allowed_feature_widths}

        if expected_feature_widths:
            got_features = int(self.channel_index.shape[1])
            if got_features not in expected_feature_widths:
                expected_str = " or ".join(str(width) for width in sorted(expected_feature_widths))
                raise ShapeError(
                    "Received incompatible sampling context feature width: "
                    f"got {got_features}, expected {expected_str}."
                )

        if num_channels is not None:
            if num_channels < 1:
                raise InvalidParameterError(f"num_channels must be >= 1, got {num_channels}.")
            active_channels = self.channel_index[self.mask]
            if active_channels.numel() > 0:
                invalid = (active_channels < 0) | (active_channels >= num_channels)
                if invalid.any():
                    invalid_values = active_channels[invalid]
                    observed_min = int(invalid_values.min().item())
                    observed_max = int(invalid_values.max().item())
                    raise InvalidParameterError(
                        "sampling_ctx.channel_index contains out-of-range channel ids on active positions: "
                        f"valid range is [0, {num_channels - 1}], observed min={observed_min}, "
                        f"max={observed_max}."
                    )

        if num_repetitions is not None:
            if num_repetitions < 1:
                raise InvalidParameterError(f"num_repetitions must be >= 1, got {num_repetitions}.")
            if self.repetition_index is None:
                if num_repetitions > 1:
                    raise InvalidParameterError(
                        "sampling_ctx.repetition_index must be provided when sampling from a module with "
                        "num_repetitions > 1."
                    )
                return

            repetition_index = self.repetition_index
            _check_repetition_index_tensor(repetition_index, name="sampling_ctx.repetition_index")
            if repetition_index.device != self.channel_index.device:
                raise InvalidParameterError(
                    "sampling_ctx.repetition_index and sampling_ctx.channel_index must be on the same device: "
                    f"got {repetition_index.device} and {self.channel_index.device}."
                )
            if repetition_index.ndim != 1:
                raise InvalidParameterError("sampling_ctx.repetition_index must have canonical shape (batch,).")

            expected_batch = int(num_samples) if num_samples is not None else int(self.channel_index.shape[0])
            if repetition_index.shape[0] != expected_batch:
                raise InvalidParameterError(
                    "sampling_ctx.repetition_index batch dimension must match the number of samples: "
                    f"got {repetition_index.shape[0]}, expected {expected_batch}."
                )

            invalid = (repetition_index < 0) | (repetition_index >= num_repetitions)
            if invalid.any():
                invalid_values = repetition_index[invalid]
                observed_min = int(invalid_values.min().item())
                observed_max = int(invalid_values.max().item())
                raise InvalidParameterError(
                    "sampling_ctx.repetition_index contains out-of-range indices: "
                    f"valid range is [0, {num_repetitions - 1}], observed min={observed_min}, "
                    f"max={observed_max}."
                )

    def __repr__(self) -> str:
        rep_shape = None if self.repetition_index is None else tuple(self.repetition_index.shape)
        return (
            "SamplingContext("
            f"channel_index.shape={tuple(self.channel_index.shape)}, "
            f"mask.shape={tuple(self.mask.shape)}, "
            f"repetition_index.shape={rep_shape}, "
            f"num_samples={self.channel_index.shape[0]}"
            ")"
        )


def update_channel_index_strict(sampling_ctx: SamplingContext, new_channel_index: Tensor) -> None:
    """Replace `channel_index` while preserving context feature-layout invariants.

    This helper is used when only channel assignments change but the feature
    layout (and therefore `mask` shape) must remain exactly the same.

    Args:
        sampling_ctx: Context to update.
        new_channel_index: New per-sample, per-feature channel assignments.

    Raises:
        ShapeError: If batch size or feature width differs from the existing
            context mask.
    """
    _check_channel_index_tensor(new_channel_index, name="new_channel_index")
    if new_channel_index.device != sampling_ctx.mask.device:
        raise InvalidParameterError(
            "new_channel_index and sampling_ctx.mask must be on the same device: "
            f"got {new_channel_index.device} and {sampling_ctx.mask.device}."
        )
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
