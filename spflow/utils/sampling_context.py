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
                If None, uses torch.get_default_device() or CPU. Defaults to None.
            channel_index (Tensor | None, optional): Channel indices for each sample
                and feature. Shape: (batch_size, num_features). If None, samples
                from all channels. Defaults to None.
            mask (Tensor | None, optional): Boolean mask indicating which instances/
                features to sample from. Must be torch.bool type. Shape must match
                channel_index if provided. Defaults to None.
            repetition_index (Tensor | None, optional): Indices for repetition-based
                sampling structures. Used by circuits with repeated computations.
                Defaults to None.

        Raises:
            InvalidParameterError: If tensor shapes are incompatible, mask has wrong dtype,
                or num_samples conflicts with tensor dimensions.
        """
        if device is None:
            # device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
            if hasattr(torch, "get_default_device"):
                device = torch.get_default_device()
            else:
                device = torch.device("cpu")

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
            # channel_index and mask must be both None or both not None
            raise InvalidParameterError("channel_index and mask must be both None or both not None.")
        elif channel_index is not None and mask is not None:
            # channel_index and mask are both not None
            _check_mask_bool(mask)
            self._mask = mask
            self._channel_index = channel_index
            self.device = device
        else:
            # channel_index and mask are both None
            if num_samples is None:
                raise InvalidParameterError(
                    "num_samples must be provided when channel_index and mask are None."
                )
            self._mask = torch.full((num_samples, 1), True, dtype=torch.bool, device=device)
            self._channel_index = torch.zeros((num_samples, 1), dtype=torch.long, device=device)
            self.device = self.mask.device

        self.repetition_idx = repetition_index

    def update(self, channel_index: Tensor, mask: Tensor):
        """Updates the sampling context with new channel index and mask.

        Args:
            channel_index: Tensor containing the channel indices to sample from.
            mask: Boolean tensor containing the mask to apply to the samples.
        """
        if not channel_index.shape == mask.shape:
            raise InvalidParameterError("channel_index and mask must have the same shape.")

        _check_mask_bool(mask)

        self._channel_index = channel_index
        self._mask = mask

    @property
    def channel_index(self):
        return self._channel_index

    @channel_index.setter
    def channel_index(self, channel_index):
        if channel_index.shape != self._mask.shape:
            raise InvalidParameterError("New channel_index and previous mask must have the same shape.")
        self._channel_index = channel_index

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask.shape != self._channel_index.shape:
            raise InvalidParameterError("New mask and previous channel_index must have the same shape.")
        _check_mask_bool(mask)
        self._mask = mask

    @property
    def samples_mask(self):
        return self.mask.sum(1) > 0

    @property
    def channel_index_masked(self):
        return self.channel_index[self.samples_mask]

    def copy(self) -> SamplingContext:
        """Return a copy of the sampling context.

        Returns:
            SamplingContext: A new SamplingContext instance with copied tensors.
        """
        return SamplingContext(
            channel_index=self.channel_index.clone(),
            mask=self.mask.clone(),
            repetition_index=self.repetition_idx.clone() if self.repetition_idx is not None else None,
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
            self.update(
                channel_index=self.channel_index.repeat(1, target_features),
                mask=self.mask.repeat(1, target_features),
            )
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

        self.update(
            channel_index=self.channel_index.repeat(1, num_splits),
            mask=self.mask.repeat(1, num_splits),
        )

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
        self.update(channel_index=channel_index, mask=mask)

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
            InvalidParameterError: If any child channel count is less than 1.
        """
        if any(count < 1 for count in child_channel_counts):
            raise InvalidParameterError("child_channel_counts must all be >= 1.")

        out: list[tuple[Tensor, Tensor]] = []
        global_channel_index = self.channel_index
        offset = 0
        for child_channels in child_channel_counts:
            child_start = offset
            child_end = offset + child_channels
            in_child_range = (global_channel_index >= child_start) & (global_channel_index < child_end)
            local_channel_index = global_channel_index - child_start
            local_channel_index = torch.where(
                in_child_range,
                local_channel_index,
                # Keep unrouted positions in-bounds for downstream gather ops.
                torch.zeros_like(local_channel_index),
            )
            child_mask = in_child_range & self.mask
            out.append((local_channel_index, child_mask))
            offset = child_end
        return out

    def __repr__(self) -> str:
        return f"SamplingContext(channel_index.shape={self.channel_index.shape}), mask.shape={self.mask.shape}), num_samples={self.channel_index.shape[0]})"


def init_default_sampling_context(
    sampling_ctx: SamplingContext | None, num_samples: int | None = None, device: torch.device | None = None
) -> SamplingContext:
    """Initializes sampling context if not already initialized.

    Args:
        sampling_ctx: SamplingContext object or None.
        num_samples: Integer specifying the number of samples.
        device: PyTorch device for tensor storage.

    Returns:
        Original sampling context if not None or a new initialized sampling context.
    """
    # Ensure, that either sampling_ctx or num_samples is not None
    if sampling_ctx is not None:
        return sampling_ctx
    else:
        return SamplingContext(num_samples=num_samples, device=device)


def build_root_sampling_context(
    sampling_ctx: SamplingContext | None,
    *,
    module_name: str,
    num_samples: int,
    num_features: int,
    device: torch.device | None = None,
) -> SamplingContext:
    """Build or validate sampling context at a root sampling entrypoint.

    Root callers should initialize a context whose feature width matches the
    root module output width so internal modules do not rely on synthetic
    feature expansion.
    """
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


def require_sampling_context(
    sampling_ctx: SamplingContext | None,
    *,
    num_samples: int | None = None,
    module_out_shape: object | None = None,
    device: torch.device | None = None,
) -> SamplingContext:
    """Validate or bootstrap sampling context for internal sampling calls.

    Internal module calls are expected to receive a context from their parent.
    The only bootstrap exception is a structural scalar node
    (`features == 1` and `channels == 1`), where creating a default context is
    shape-safe and does not require routing decisions.

    Args:
        sampling_ctx: Existing context from the parent call.
        num_samples: Expected batch size for validation.
        module_out_shape: Shape-like object used only for bootstrap eligibility.
        device: Device used when bootstrap allocation is required.

    Returns:
        A validated or newly created `SamplingContext`.

    Raises:
        InvalidParameterError: If context is missing for non-bootstrapable
            modules, or if batch dimensions mismatch.
    """
    if sampling_ctx is None:
        # Bootstrapping is intentionally narrow to avoid hiding routing bugs.
        can_bootstrap = (
            module_out_shape is not None
            and hasattr(module_out_shape, "features")
            and hasattr(module_out_shape, "channels")
            and int(module_out_shape.features) == 1
            and int(module_out_shape.channels) == 1
        )
        if can_bootstrap:
            if num_samples is None:
                raise InvalidParameterError(
                    "Cannot initialize sampling context without num_samples."
                )
            return SamplingContext(num_samples=num_samples, device=device)

        raise InvalidParameterError(
            "Sampling requires an explicit sampling_ctx for internal sampling unless "
            "module.out_shape.features == 1 and module.out_shape.channels == 1."
        )

    if num_samples is not None and sampling_ctx.channel_index.shape[0] != num_samples:
        raise InvalidParameterError(
            "sampling_ctx batch size mismatch: "
            f"got {sampling_ctx.channel_index.shape[0]}, "
            f"expected {num_samples}."
        )

    return sampling_ctx


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

