from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, ScopeError, ShapeError
from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import (
    DifferentiableSamplingContext,
    SamplingContext,
)


class BaseProduct(Module, ABC):
    """Base class for product operations in probabilistic circuits.

    Computes joint distributions via factorization. Assumes conditional independence
    between disjoint input scopes. Abstract class requiring mapping method implementations.

    Attributes:
        inputs: Input modules to multiply.
        input_is_split: Whether input is a split operation.
        num_splits: Number of splits if input is split.
        scope: Combined scope of all inputs.
    """

    def __init__(
        self,
        inputs: list[Module] | Module,
    ) -> None:
        """Initialize product module.

        Args:
            inputs: Input module(s) with pairwise disjoint scopes.

        Raises:
            ValueError: No inputs provided.
            ScopeError: Input scopes not pairwise disjoint.
        """
        super().__init__()

        # ========== 1. INPUT VALIDATION ==========
        # Validate before processing to avoid indexing errors
        if isinstance(inputs, list) and not inputs:
            raise ValueError(f"'{self.__class__.__name__}' requires at least one input to be specified.")

        # ========== 2. INPUT TYPE PROCESSING ==========
        # Check if input is Split type and handle accordingly
        if isinstance(inputs, Split):
            self.input_is_split = True
            self.num_splits = inputs.num_splits
            inputs = [inputs]
        else:
            self.input_is_split = False
            # Determine num_splits from first input's features
            if inputs[0].out_shape.features == 1:
                self.num_splits = 1
            else:
                self.num_splits = None

        # ========== 3. CONFIGURATION VALIDATION ==========
        # Validate scopes are pairwise disjoint
        if not Scope.all_pairwise_disjoint([inp.scope for inp in inputs]):
            raise ScopeError("Input scopes must be disjoint.")

        # ========== 4. INPUT MODULE SETUP ==========
        self.inputs = nn.ModuleList(inputs)

        # ========== 5. ATTRIBUTE INITIALIZATION ==========
        # Join all input scopes to create combined scope
        self.scope = Scope.join_all([inp.scope for inp in self.inputs])

        # Set in_shape early so subclasses can use in_shape.channels
        # in_channels = max channels across all inputs (for broadcasting)
        in_channels = max(inp.out_shape.channels for inp in self.inputs)
        self.in_shape = ModuleShape(
            self.inputs[0].out_shape.features, in_channels, self.inputs[0].out_shape.repetitions
        )
        # Note: out_shape must be set by subclasses

    @abstractmethod
    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        """Map output channel indices to input channel indices.

        Args:
            output_ids: Tensor containing output channel indices.

        Returns:
            Tensor: Mapped input channel indices.
        """
        pass

    @abstractmethod
    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        """Map output mask to input mask.

        Args:
            mask: Output mask tensor.

        Returns:
            Tensor: Mapped input mask tensor.
        """
        pass

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}"

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        """Generate samples from product module.

        Args:
            num_samples: Number of samples to generate.
            data: Optional data tensor to store samples.
            is_mpe: Whether to perform most probable explanation.
            cache: Optional cache for computation.
            sampling_ctx: Optional sampling context.

        Returns:
            Tensor: Generated samples.
        """
        # Prepare data tensor

        # initialize contexts
        sampling_ctx.require_feature_width(expected_features=self.out_shape.features)

        # Map to (i, j) to index left/right inputs
        channel_index = self.map_out_channels_to_in_channels(sampling_ctx.channel_index)
        mask = self.map_out_mask_to_in_mask(sampling_ctx.mask)

        cid_per_module = []
        mask_per_module = []

        inputs = self.inputs
        for i in range(len(self.inputs)):
            cid_per_module.append(channel_index[..., i])
            mask_per_module.append(mask[..., i])

        # Iterate over inputs, their channel indices and masks
        for inp, cid, mask in zip(inputs, cid_per_module, mask_per_module):
            if cid.ndim == 1:
                cid = cid.unsqueeze(1)
            if mask.ndim == 1:
                mask = mask.unsqueeze(1)
            sampling_ctx.update(channel_index=cid, mask=mask)
            inp._sample(
                data=data,
                is_mpe=is_mpe,
                cache=cache,
                sampling_ctx=sampling_ctx,
            )

        return data

    def _rsample(
        self,
        data: Tensor,
        sampling_ctx: DifferentiableSamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        """Generate differentiable samples from product module via probability routing."""
        sampling_ctx.require_feature_width(expected_features=self.out_shape.features)
        parent_channel_probs = sampling_ctx.resolve_channel_probs(
            expected_channels=int(self.out_shape.channels),
            module_name=f"{self.__class__.__name__}._rsample",
        )

        num_output_features = int(self.out_shape.features)
        num_output_channels = int(self.out_shape.channels)
        num_inputs = len(self.inputs)

        output_channel_ids = torch.arange(
            num_output_channels, dtype=torch.long, device=sampling_ctx.channel_probs.device
        ).unsqueeze(1)
        output_channel_ids = output_channel_ids.expand(num_output_channels, num_output_features)
        mapped_channel_ids = self.map_out_channels_to_in_channels(output_channel_ids)

        if mapped_channel_ids.ndim != 3 or int(mapped_channel_ids.shape[2]) != num_inputs:
            raise ShapeError(
                f"{self.__class__.__name__}.map_out_channels_to_in_channels must return "
                f"shape (batch, features, {num_inputs}), got {tuple(mapped_channel_ids.shape)}."
            )

        output_feature_basis = torch.eye(
            num_output_features,
            dtype=torch.bool,
            device=sampling_ctx.channel_probs.device,
        )
        mapped_feature_masks = self.map_out_mask_to_in_mask(output_feature_basis)
        if mapped_feature_masks.ndim != 3 or int(mapped_feature_masks.shape[2]) != num_inputs:
            raise ShapeError(
                f"{self.__class__.__name__}.map_out_mask_to_in_mask must return "
                f"shape (batch, features, {num_inputs}), got {tuple(mapped_feature_masks.shape)}."
            )
        if mapped_feature_masks.shape[1] != mapped_channel_ids.shape[1]:
            raise ShapeError(
                f"{self.__class__.__name__} produced incompatible mapped feature widths in _rsample: "
                f"mask width {mapped_feature_masks.shape[1]} vs channel-id width {mapped_channel_ids.shape[1]}."
            )

        parent_mask = sampling_ctx.mask

        child_channel_probs_per_module: list[Tensor] = []
        child_mask_per_module: list[Tensor] = []
        mapped_input_features = int(mapped_channel_ids.shape[1])

        for input_idx, input_module in enumerate(self.inputs):
            child_channels = int(input_module.out_shape.channels)
            child_channel_probs = parent_channel_probs.new_zeros(
                (parent_channel_probs.shape[0], mapped_input_features, child_channels)
            )
            child_mask_mass = parent_channel_probs.new_zeros(
                (parent_channel_probs.shape[0], mapped_input_features)
            )

            for output_feature_idx in range(num_output_features):
                target_features = mapped_feature_masks[output_feature_idx, :, input_idx]
                if not bool(target_features.any().item()):
                    continue

                parent_feature_probs = parent_channel_probs[:, output_feature_idx, :]
                parent_feature_mask = parent_mask[:, output_feature_idx]

                if child_channels == 1:
                    local_channel_ids = torch.zeros(
                        (num_output_channels, int(target_features.sum().item())),
                        dtype=torch.long,
                        device=parent_channel_probs.device,
                    )
                else:
                    local_channel_ids = mapped_channel_ids[:, target_features, input_idx].to(dtype=torch.long)
                    invalid = (local_channel_ids < 0) | (local_channel_ids >= child_channels)
                    if invalid.any():
                        invalid_values = local_channel_ids[invalid]
                        observed_min = int(invalid_values.min().item())
                        observed_max = int(invalid_values.max().item())
                        raise InvalidParameterError(
                            f"{self.__class__.__name__} produced invalid mapped child channel ids "
                            f"for child {input_idx}: expected range [0, {child_channels - 1}], "
                            f"observed min={observed_min}, max={observed_max}."
                        )

                channel_assignments = F.one_hot(local_channel_ids, num_classes=child_channels).to(
                    dtype=parent_feature_probs.dtype
                )
                contribution = torch.einsum("bc,cfk->bfk", parent_feature_probs, channel_assignments)
                contribution = contribution * parent_feature_mask.to(dtype=contribution.dtype).unsqueeze(
                    -1
                ).unsqueeze(-1)

                child_channel_probs[:, target_features, :] = (
                    child_channel_probs[:, target_features, :] + contribution
                )
                child_mask_mass[:, target_features] = child_mask_mass[
                    :, target_features
                ] + parent_feature_mask.to(dtype=child_mask_mass.dtype).unsqueeze(-1)

            child_mask = child_mask_mass > 0.0
            norm = child_channel_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            child_channel_probs = torch.where(
                child_mask.unsqueeze(-1),
                child_channel_probs / norm,
                torch.zeros_like(child_channel_probs),
            )

            child_channel_probs_per_module.append(child_channel_probs)
            child_mask_per_module.append(child_mask)

        for input_module, child_channel_probs, child_mask in zip(
            self.inputs, child_channel_probs_per_module, child_mask_per_module
        ):
            sampling_ctx.update_prob_routing(
                channel_probs=child_channel_probs,
                mask=child_mask,
            )
            input_module._rsample(
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
        cache: Cache | None = None,
    ) -> Optional["BaseProduct | Module"]:
        """Marginalize specified variables (must be implemented by subclasses).

        Args:
            marg_rvs: List of variable indices to marginalize.
            prune: Whether to prune the resulting structure.
            cache: Optional cache for computation.

        Returns:
            Optional[BaseProduct | Module]: Marginalized module or None.
        """
        # This is not yet implemented for BaseProduct
        # Reasons: Marginalization over the element-product has a couple of challenges:
        # - If the input is a single module, we need to ensure the splits are still equally sized
        # - We need to eventually update the split_indices (non-trivial mapping)
        # - If the input are two modules, we need to ensure both inputs have the same number of features
        raise NotImplementedError(
            "Not implemented for BaseProduct -- needs to be called on subclasses of BaseProduct."
        )

    @abstractmethod
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood.

        Args:
            data: Input data tensor.
            cache: Optional cache for computation.

        Returns:
            Tensor: Log likelihood values.
        """
        pass

    def _get_input_log_likelihoods(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> list[Tensor]:
        """Prepare input log-likelihoods.

        Args:
            data: Input data tensor.
            cache: Optional cache for computation.

        Returns:
            list[Tensor]: List of log-likelihood tensors for each input.
        """
        if self.input_is_split:
            lls = self.inputs[0].log_likelihood(
                data,
                cache=cache,
            )

        else:
            lls = []
            for inp in self.inputs:
                ll = inp.log_likelihood(
                    data,
                    cache=cache,
                )
                lls.append(ll)

        return lls
