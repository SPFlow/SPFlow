"""WeightedSum module for non-normalized (possibly sparse) weights.

This module is used for QPC materialization where weights can represent:
- quadrature weights (not normalized), and/or
- structural sparsity (e.g., Eq. (4) in PICs yields block-diagonal matrices with zeros).
"""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from spflow.exceptions import InvalidWeightsError, ShapeError
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.cat import Cat
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import (
    SamplingContext,
    update_channel_index_strict,
)


class WeightedSum(Module):
    """Sum module with non-normalized weights for quadrature integration.

    Unlike the standard Sum module which normalizes weights via softmax,
    WeightedSum preserves exact weight values. This is essential for
    Quadrature Probabilistic Circuits (QPCs) where weights represent
    integration weights from numerical quadrature.

    Attributes:
        inputs (Module): Input module(s) to the sum node.
        weights (Parameter): Raw (non-normalized) weights tensor.
    """

    def __init__(
        self,
        inputs: Module | list[Module],
        weights: Tensor,
        num_repetitions: int = 1,
    ) -> None:
        """Create a WeightedSum module with explicit weights.

        Args:
            inputs: Single module or list of modules to weight.
            weights: Weight tensor. Shape should be compatible with
                (features, in_channels, out_channels, repetitions).
            num_repetitions: Number of repetitions for structured representations.

        Raises:
            ValueError: If inputs empty or weights have invalid shape.
        """
        super().__init__()

        # ========== 1. INPUT VALIDATION ==========
        if not inputs:
            raise ValueError("'WeightedSum' requires at least one input to be specified.")

        # ========== 3. INPUT MODULE SETUP ==========
        if isinstance(inputs, list):
            if len(inputs) == 1:
                self.inputs = inputs[0]
            else:
                self.inputs = Cat(inputs=inputs, dim=2)
        else:
            self.inputs = inputs

        self.sum_dim = 1
        self.scope = self.inputs.scope

        # ========== 4. SHAPE COMPUTATION ==========
        self.in_shape = self.inputs.out_shape

        # ========== 5. PROCESS + VALIDATE WEIGHTS ==========
        if isinstance(weights, (list, tuple)):
            weights = torch.as_tensor(weights, dtype=torch.get_default_dtype())

        if weights.dim() == 1:
            weights = rearrange(weights, "ci -> 1 ci 1 1")
        elif weights.dim() == 2:
            weights = rearrange(weights, "ci co -> 1 ci co 1")
        elif weights.dim() == 3:
            weights = rearrange(weights, "f ci co -> f ci co 1")
        elif weights.dim() == 4:
            pass
        else:
            raise ShapeError(
                f"Weights for 'WeightedSum' must be 1D, 2D, 3D, or 4D tensor but was {weights.dim()}D."
            )

        if not torch.all(weights >= 0):
            raise InvalidWeightsError("Weights for 'WeightedSum' must be non-negative.")

        # Allow broadcasting weights across features, but not across channels/repetitions.
        if weights.shape[0] == 1 and self.in_shape.features > 1:
            weights = repeat(weights, "1 ci co r -> f ci co r", f=self.in_shape.features)

        if weights.shape[0] != self.in_shape.features:
            raise ShapeError(
                f"Weights first dimension must match number of features ({self.in_shape.features}) or be 1, "
                f"but was {weights.shape[0]}."
            )
        if weights.shape[1] != self.in_shape.channels:
            raise ShapeError(
                f"Weights in_channels dimension must match input channels ({self.in_shape.channels}), "
                f"but was {weights.shape[1]}."
            )

        out_channels = weights.shape[2]
        num_repetitions = weights.shape[3]

        self.out_shape = ModuleShape(
            features=self.in_shape.features, channels=out_channels, repetitions=num_repetitions
        )

        # ========== 6. WEIGHT REGISTRATION ==========
        self._weights = nn.Parameter(weights)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.inputs.feature_to_scope

    @property
    def weights(self) -> Tensor:
        """Returns the raw (non-normalized) weights tensor.

        Returns:
            Tensor: Weights as stored, without normalization.
        """
        return self._weights

    @weights.setter
    def weights(self, values: Tensor) -> None:
        """Set weights directly (no normalization applied).

        Args:
            values: Weight tensor with shape (features, in_channels, out_channels, repetitions).
        """
        if values.shape != self._weights.shape:
            raise ShapeError(
                f"Invalid shape for weights: Was {values.shape} but expected {self._weights.shape}."
            )
        self._weights.data = values

    @property
    def log_weights(self) -> Tensor:
        """Returns the log weights (log of raw weights).

        Returns:
            Tensor: Log of weights, no softmax applied.
        """
        neg_inf = torch.full_like(self._weights, float("-inf"))
        return torch.where(self._weights > 0, torch.log(self._weights), neg_inf)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, weights={tuple(self._weights.shape)}"

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood P(data | module).

        Uses logsumexp for numerical stability with the stored (non-normalized) weights.

        Args:
            data: Input data of shape (batch_size, num_features).
            cache: Cache for intermediate computations. Defaults to None.

        Returns:
            Tensor: Log-likelihood of shape (batch_size, num_features, out_channels, repetitions).
        """
        # Get input log-likelihoods
        ll = self.inputs.log_likelihood(data, cache=cache)

        ll = rearrange(ll, "b f ci r -> b f ci 1 r")

        log_weights = rearrange(self.log_weights, "f ci co r -> 1 f ci co r")

        # Weighted log-likelihoods
        weighted_lls = ll + log_weights  # shape: (B, F, IC, OC, R)

        # Sum over input channels (sum_dim + 1 since batch dimension is first)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1)

        return output

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        """Generate samples from WeightedSum module.

        Args:
            num_samples: Number of samples to generate.
            data: Data tensor with NaN values to fill with samples.
            is_mpe: Whether to perform maximum a posteriori estimation.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Tensor: Sampled values.
        """
        sampling_ctx.validate_sampling_context(
            num_samples=data.shape[0],
            num_features=self.out_shape.features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, self.out_shape.features),
        )
        sampling_ctx.broadcast_feature_width(target_features=self.out_shape.features, allow_from_one=True)

        # Use weights directly (not logits)
        weights = self._weights

        # Index into the correct weight channels given by parent module
        if sampling_ctx.repetition_index is not None:
            batch_size = int(sampling_ctx.channel_index.shape[0])
            weights = repeat(weights, "f ci co r -> b f ci co r", b=batch_size)
            num_features = int(weights.shape[1])
            num_input_channels = int(weights.shape[2])
            num_output_channels = int(weights.shape[3])
            indices = repeat(
                rearrange(sampling_ctx.repetition_index, "... -> (...)"),
                "b -> b f ci co 1",
                f=num_features,
                ci=num_input_channels,
                co=num_output_channels,
            )
            weights = torch.gather(weights, dim=-1, index=indices)
            weights = rearrange(weights, "b f ci co 1 -> b f ci co")
        else:
            if self.out_shape.repetitions > 1:
                raise ValueError(
                    "sampling_ctx.repetition_index must be provided when sampling from a module with "
                    "num_repetitions > 1."
                )
            batch_size = int(sampling_ctx.channel_index.shape[0])
            weights = repeat(weights[..., 0], "f ci co -> b f ci co", b=batch_size)

        in_channels_total = weights.shape[2]
        idxs = repeat(sampling_ctx.channel_index, "b f -> b f ci 1", ci=in_channels_total)
        weights = weights.gather(dim=3, index=idxs)
        weights = rearrange(weights, "b f ci 1 -> b f ci")

        # Sample from categorical distribution
        if sampling_ctx.is_mpe:
            new_channel_index = torch.argmax(weights, dim=-1)
        else:
            # Normalize for sampling (temporary normalization for distribution)
            denom = weights.sum(dim=-1, keepdim=True)
            invalid_rows = (denom <= 0).squeeze(-1)
            if invalid_rows.any():
                num_invalid_rows = int(invalid_rows.sum().item())
                raise ShapeError(
                    "WeightedSum.sample encountered zero-sum routing weights for "
                    f"{num_invalid_rows} feature rows. Sampling is undefined for these rows."
                )
            probs = weights / denom
            new_channel_index = torch.distributions.Categorical(probs=probs).sample()

        update_channel_index_strict(sampling_ctx, new_channel_index)

        # Sample from input module
        self.inputs._sample(
            data=data,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

        return data

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> WeightedSum | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variables to marginalize.
            prune: Whether to prune the module.
            cache: Optional cache dictionary.

        Returns:
            Marginalized WeightedSum module or None.
        """
        module_scope = self.scope
        marg_input = None

        mutual_rvs = set(module_scope.query).intersection(set(marg_rvs))
        module_weights = self._weights.data.clone()

        # Module scope is being fully marginalized over
        if len(mutual_rvs) == len(module_scope.query):
            return None

        # Node scope is being partially marginalized
        elif mutual_rvs:
            marg_input = self.inputs.marginalize(marg_rvs, prune=prune, cache=cache)

            if marg_input:
                # Apply mask to weights per-repetition
                masked_weights_list = []
                for r in range(self.out_shape.repetitions):
                    feature_to_scope_r = self.inputs.feature_to_scope[:, r].copy()
                    for rv in mutual_rvs:
                        for idx, scope in enumerate(feature_to_scope_r):
                            if scope is not None:
                                if rv in scope.query:
                                    feature_to_scope_r[idx] = scope.remove_from_query(rv)

                    mask = torch.tensor(
                        [not scope.empty() for scope in feature_to_scope_r], device=self.device
                    ).bool()

                    masked_weights_r = module_weights[:, :, :, r][mask]
                    masked_weights_list.append(masked_weights_r)

                if all(w.shape[0] == masked_weights_list[0].shape[0] for w in masked_weights_list):
                    module_weights = torch.stack(masked_weights_list, dim=-1)
                else:
                    max_features = max(w.shape[0] for w in masked_weights_list)
                    padded_list = []
                    for w in masked_weights_list:
                        if w.shape[0] < max_features:
                            padding = torch.zeros(
                                max_features - w.shape[0],
                                w.shape[1],
                                w.shape[2],
                                device=w.device,
                                dtype=w.dtype,
                            )
                            w = torch.cat([w, padding], dim=0)
                        padded_list.append(w)
                    module_weights = torch.stack(padded_list, dim=-1)
        else:
            marg_input = self.inputs

        if marg_input is None:
            return None
        else:
            return WeightedSum(inputs=marg_input, weights=module_weights)
