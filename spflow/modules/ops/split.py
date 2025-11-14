"""Split operations for tensor partitioning in probabilistic circuits.

Provides base classes and implementations for splitting tensors along
dimensions. Essential for RAT-SPNs and other architectures requiring
systematic tensor partitioning.
"""

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Any, Dict, Optional

from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.utils.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)


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
        self.inputs = nn.ModuleList([inputs])

        self.dim = dim
        self.num_splits = num_splits
        self.num_repetitions = self.inputs[0].num_repetitions
        self.scope = self.inputs[0].scope

    @property
    def out_features(self) -> int:
        return self.inputs[0].out_features

    @property
    def out_channels(self) -> int:
        return self.inputs[0].out_channels

    def get_out_shapes(self, event_shape):
        """Get output shapes for each split based on input event shape."""
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
    def feature_to_scope(self) -> list[Scope]:
        pass

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Optional[Dict[str, Any]] = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples by delegating to input module."""
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        # initialize context
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        # Expand mask and channels to match input module shape
        mask = sampling_ctx.mask.expand(data.shape[0], self.inputs[0].out_features)
        channel_index = sampling_ctx.channel_index.expand(data.shape[0], self.inputs[0].out_features)
        sampling_ctx.update(channel_index=channel_index, mask=mask)

        self.inputs[0].sample(
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
        """Marginalize out specified random variables."""
        # compute module scope (same for all outputs)
        module_scope = self.scope

        mutual_rvs = set(module_scope.query).intersection(set(marg_rvs))

        # Node scope is only being partially marginalized
        if mutual_rvs:
            # marginalize child modules

            marg_child_module = self.inputs[0].marginalize(marg_rvs, prune=prune, cache=cache)

            # if marginalized child is not None
            if marg_child_module:
                if prune and marg_child_module.out_features == 1:
                    return marg_child_module
                else:
                    return self.__class__(inputs=marg_child_module, dim=self.dim, num_splits=self.num_splits)

            # if all children were marginalized, return None
            else:
                return None

            # if only a single input survived marginalization, return it if pruning is enabled
        else:
            return self
