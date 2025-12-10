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

        if not isinstance(inputs, Module):
            raise ValueError(f"'{self.__class__.__name__}' requires a single Module as input.")

        self.inputs = inputs

        self.dim = dim
        self.num_splits = num_splits
        self.num_repetitions = self.inputs.num_repetitions
        self.scope = self.inputs.scope

        # Note: _infer_shapes() not called here because subclasses may need
        # to complete their initialization first

    def _infer_shapes(self) -> None:
        """Compute and set input/output shapes for Split module."""
        from spflow.modules.module_shape import ModuleShape

        self._input_shape = self.inputs.output_shape
        self._output_shape = ModuleShape(
            self.out_features, self.out_channels, self.num_repetitions
        )


    @property
    def out_features(self) -> int:
        return self.inputs.out_features

    @property
    def out_channels(self) -> int:
        return self.inputs.out_channels

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
        mask = sampling_ctx.mask.expand(data.shape[0], self.inputs.out_features)
        channel_index = sampling_ctx.channel_index.expand(data.shape[0], self.inputs.out_features)
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
