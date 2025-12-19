"""Split operations for tensor partitioning in probabilistic circuits.

Provides base classes and implementations for splitting tensors along
dimensions. Essential for RAT-SPNs and other architectures requiring
systematic tensor partitioning.
"""

from __future__ import annotations
import numpy as np

from abc import abstractmethod, ABC
from typing import Any, Dict, Optional

from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)


class SplitMode:
    """Configuration for split operations.

    Factory class for creating split configurations. Use the class methods
    to create split configurations that can be passed to modules.

    Example:
        >>> layer = EinsumLayer(inputs=leaf, split_mode=SplitMode.interleaved(num_splits=3))
        >>> layer = LinsumLayer(inputs=leaf, split_mode=SplitMode.consecutive(num_splits=2))
        >>> layer = LinsumLayer(inputs=leaf, split_mode=SplitMode.by_index(indices=[[0,1], [2,3]]))
    """

    def __init__(self, split_type: str, num_splits: int = 2, indices: list[list[int]] | None = None):
        """Initialize split mode configuration.

        Args:
            split_type: Type of split ('consecutive', 'interleaved', or 'by_index').
            num_splits: Number of parts to split into.
            indices: For 'by_index' type, the feature indices for each split.
        """
        if split_type not in ('consecutive', 'interleaved', 'by_index'):
            raise ValueError(f"split_type must be 'consecutive', 'interleaved', or 'by_index', got '{split_type}'")
        if split_type != 'by_index' and num_splits < 2:
            raise ValueError(f"num_splits must be at least 2, got {num_splits}")
        if split_type == 'by_index' and indices is None:
            raise ValueError("indices must be provided for 'by_index' split type")

        self._split_type = split_type
        self._num_splits = num_splits
        self._indices = indices

    @property
    def num_splits(self) -> int:
        """Number of splits."""
        return self._num_splits

    @property
    def split_type(self) -> str:
        """Type of split ('consecutive', 'interleaved', or 'by_index')."""
        return self._split_type

    @property
    def indices(self) -> list[list[int]] | None:
        """Feature indices for 'by_index' split type."""
        return self._indices

    @classmethod
    def consecutive(cls, num_splits: int = 2) -> "SplitMode":
        """Create a consecutive split configuration.

        Splits features into consecutive chunks: [0,1,2,3] -> [0,1], [2,3].

        Args:
            num_splits: Number of parts to split into.

        Returns:
            SplitMode configuration for consecutive splitting.
        """
        return cls('consecutive', num_splits)

    @classmethod
    def interleaved(cls, num_splits: int = 2) -> "SplitMode":
        """Create an interleaved split configuration.

        Splits features using modulo: [0,1,2,3] -> [0,2], [1,3].

        Args:
            num_splits: Number of parts to split into.

        Returns:
            SplitMode configuration for interleaved splitting.
        """
        return cls('interleaved', num_splits)

    @classmethod
    def by_index(cls, indices: list[list[int]]) -> "SplitMode":
        """Create a split configuration with explicit feature indices.

        Splits features according to specified indices. Each inner list
        contains the feature indices for that split.

        Example:
            >>> SplitMode.by_index([[0, 1, 4], [2, 3, 5, 6, 7]])
            # Creates 2 splits: features [0,1,4] and features [2,3,5,6,7]

        Args:
            indices: List of lists specifying feature indices for each split.
                All features must be covered exactly once.

        Returns:
            SplitMode configuration for index-based splitting.
        """
        return cls('by_index', num_splits=len(indices), indices=indices)

    def create(self, inputs: Module) -> "Split":
        """Create a Split module with this configuration.

        Args:
            inputs: Input module to split.

        Returns:
            Split module configured according to this SplitMode.
        """
        # Import here to avoid circular imports
        from spflow.modules.ops.split_consecutive import SplitConsecutive
        from spflow.modules.ops.split_interleaved import SplitInterleaved
        from spflow.modules.ops.split_by_index import SplitByIndex

        if self._split_type == 'consecutive':
            return SplitConsecutive(inputs, num_splits=self._num_splits)
        elif self._split_type == 'interleaved':
            return SplitInterleaved(inputs, num_splits=self._num_splits)
        else:  # by_index
            return SplitByIndex(inputs, indices=self._indices)

    def __repr__(self) -> str:
        if self._split_type == 'by_index':
            return f"SplitMode.by_index(indices={self._indices})"
        return f"SplitMode.{self._split_type}(num_splits={self._num_splits})"


class Split(Module, ABC):
    """Abstract base class for tensor splitting operations.

    Splits input tensors along specified dimensions. Concrete implementations
    must provide feature_to_scope property.

    Attributes:
        inputs (nn.ModuleList): Single input module to split.
        dim (int): Dimension along which to split (0=batch, 1=feature, 2=channel).
        num_splits (int): Number of splits to create.
        scope (Scope): Variable scope inherited from input.
    """

    def __init__(self, inputs: Module, dim: int = 1, num_splits: int | None = 2):
        """Initialize split operation.

        Args:
            inputs: Input module to split.
            dim: Dimension along which to split (0=batch, 1=feature, 2=channel).
            num_splits: Number of parts to split into.
        """
        super().__init__()

        if not isinstance(inputs, Module):
            raise ValueError(f"'{self.__class__.__name__}' requires a single Module as input.")

        self.inputs = inputs

        self.dim = dim
        self.num_splits = num_splits
        self.scope = self.inputs.scope

        # Shape computation
        in_shape = self.inputs.out_shape
        self.in_shape = in_shape
        self.out_shape = ModuleShape(
            in_shape.features, in_shape.channels, in_shape.repetitions
        )


    def get_out_shapes(self, event_shape):
        """Get output shapes for each split based on input event shape.

        Args:
            event_shape: Shape of the input event tensor.

        Returns:
            List of tuples representing output shapes for each split.
        """
        split_size = event_shape[self.dim]
        quotient = split_size // self.num_splits
        remainder = split_size % self.num_splits
        if self.dim == 0:
            if remainder == 0:
                return [(quotient, event_shape[1])] * self.num_splits
            else:
                return [(quotient, event_shape[1])] * (self.num_splits - 1) + [(remainder, event_shape[1])]

        else:
            if remainder == 0:
                return [(event_shape[0], quotient)] * self.num_splits
            else:
                return [(event_shape[0], quotient)] * (self.num_splits - 1) + [(event_shape[1], remainder)]

    @property
    @abstractmethod
    def feature_to_scope(self) -> np.ndarray:
        pass

    @abstractmethod
    def merge_split_indices(self, *split_indices: Tensor) -> Tensor:
        """Merge per-split channel indices back to original feature layout.

        This method takes channel indices for each split and combines them into
        indices matching the original (unsplit) feature layout. Used by parent
        modules (like EinsumLayer) during sampling.

        Args:
            *split_indices: Channel index tensors for each split, shape (batch, features_per_split).

        Returns:
            Merged indices matching the input module's feature layout, shape (batch, total_features).
        """
        pass

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Optional[Dict[str, Any]] = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples by delegating to input module.

        Args:
            num_samples: Number of samples to generate.
            data: Existing data tensor to modify.
            is_mpe: Whether to perform most probable explanation.
            cache: Cache dictionary for intermediate results.
            sampling_ctx: Sampling context for controlling sample generation.

        Returns:
            Tensor containing the generated samples.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        # initialize context
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        # Expand mask and channels to match input module shape
        mask = sampling_ctx.mask.expand(data.shape[0], self.inputs.out_shape.features)
        channel_index = sampling_ctx.channel_index.expand(data.shape[0], self.inputs.out_shape.features)
        sampling_ctx.update(channel_index=channel_index, mask=mask)

        self.inputs.sample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )
        return data

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Optional[Dict[str, Any]] = None,
    ) -> None | Module:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variable indices to marginalize.
            prune: Whether to prune the resulting module.
            cache: Cache dictionary for intermediate results.

        Returns:
            Marginalized module or None if fully marginalized.
        """
        # compute module scope (same for all outputs)
        module_scope = self.scope

        mutual_rvs = set(module_scope.query).intersection(set(marg_rvs))

        # Node scope is only being partially marginalized
        if mutual_rvs:
            # marginalize child modules

            marg_child_module = self.inputs.marginalize(marg_rvs, prune=prune, cache=cache)

            # if marginalized child is not None
            if marg_child_module:
                if prune and marg_child_module.out_shape.features == 1:
                    return marg_child_module
                else:
                    return self.__class__(inputs=marg_child_module, dim=self.dim, num_splits=self.num_splits)

            # if all children were marginalized, return None
            else:
                return None

            # if only a single input survived marginalization, return it if pruning is enabled
        else:
            return self
