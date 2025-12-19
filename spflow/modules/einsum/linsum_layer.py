"""LinsumLayer for efficient linear sum-product operations in probabilistic circuits.

Unlike EinsumLayer which computes a cross-product of input channels,
LinsumLayer computes a linear combination: it adds left/right features
(product in log-space), then applies a weighted sum over input channels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.split import Split, SplitMode
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.utils.cache import Cache, cached
from spflow.utils.projections import proj_convex_to_real
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class LinsumLayer(Module):
    """LinsumLayer combining product and sum operations with linear channel combination.

    Unlike EinsumLayer which computes cross-product over channels (I × J combinations),
    LinsumLayer computes a linear combination: pairs left/right features, adds them
    (product in log-space), then sums over input channels with learned weights.

    This results in fewer parameters: weight_shape = (D_out, O, R, C) vs
    EinsumLayer's (D_out, O, R, I, J).

    Attributes:
        logits (Parameter): Unnormalized log-weights for gradient optimization.
    """

    def __init__(
        self,
        inputs: Module | list[Module],
        out_channels: int,
        num_repetitions: int | None = None,
        weights: Tensor | None = None,
        split_mode: SplitMode | None = None,
    ) -> None:
        """Initialize LinsumLayer.

        Args:
            inputs: Either a single module (features will be split into pairs)
                or a list of exactly two modules (left and right children).
                Unlike EinsumLayer, both inputs must have the same number of channels.
            out_channels: Number of output sum nodes per feature.
            num_repetitions: Number of repetitions. If None, inferred from inputs.
            weights: Optional initial weights tensor. If provided, must have shape
                (out_features, out_channels, num_repetitions, in_channels).
            split_mode: Optional split configuration for single input mode.
                Use SplitMode.consecutive() or SplitMode.interleaved().
                Defaults to SplitMode.consecutive(num_splits=2) if not specified.

        Raises:
            ValueError: If inputs invalid, out_channels < 1, or weight shape mismatch.
        """
        super().__init__()

        # ========== 1. INPUT VALIDATION ==========
        if isinstance(inputs, list):
            if len(inputs) != 2:
                raise ValueError(
                    f"LinsumLayer requires exactly 2 input modules when given a list, got {len(inputs)}."
                )
            self._two_inputs = True
            left_input, right_input = inputs

            # LinsumLayer requires same number of channels (linear combination, not cross-product)
            if left_input.out_shape.channels != right_input.out_shape.channels:
                raise ValueError(
                    f"LinsumLayer requires left and right inputs to have same number of channels: "
                    f"{left_input.out_shape.channels} != {right_input.out_shape.channels}"
                )
            if left_input.out_shape.features != right_input.out_shape.features:
                raise ValueError(
                    f"Left and right inputs must have same number of features: "
                    f"{left_input.out_shape.features} != {right_input.out_shape.features}"
                )
            if left_input.out_shape.repetitions != right_input.out_shape.repetitions:
                raise ValueError(
                    f"Left and right inputs must have same number of repetitions: "
                    f"{left_input.out_shape.repetitions} != {right_input.out_shape.repetitions}"
                )
            # Validate disjoint scopes
            if not Scope.all_pairwise_disjoint([left_input.scope, right_input.scope]):
                raise ValueError("Left and right input scopes must be disjoint.")

            self.inputs = nn.ModuleList([left_input, right_input])
            in_channels = left_input.out_shape.channels
            in_features = left_input.out_shape.features
            if num_repetitions is None:
                num_repetitions = left_input.out_shape.repetitions
            self.scope = Scope.join_all([left_input.scope, right_input.scope])

        else:
            # Single input: will split features into left/right halves
            self._two_inputs = False
            if inputs.out_shape.features < 2:
                raise ValueError(
                    f"LinsumLayer requires at least 2 input features for splitting, "
                    f"got {inputs.out_shape.features}."
                )
            if inputs.out_shape.features % 2 != 0:
                raise ValueError(
                    f"LinsumLayer requires even number of input features for splitting, "
                    f"got {inputs.out_shape.features}."
                )
            # Use Split directly if already a split module, otherwise create from split_mode
            if isinstance(inputs, Split):
                self.inputs = inputs
            elif split_mode is not None:
                self.inputs = split_mode.create(inputs)
            else:
                # Default: consecutive split with 2 parts
                self.inputs = SplitConsecutive(inputs)
            in_channels = inputs.out_shape.channels
            in_features = inputs.out_shape.features // 2
            if num_repetitions is None:
                num_repetitions = inputs.out_shape.repetitions
            self.scope = inputs.scope

        # ========== 2. CONFIGURATION VALIDATION ==========
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}.")

        # ========== 3. SHAPE COMPUTATION ==========
        self._in_channels = in_channels
        self.in_shape = ModuleShape(in_features, in_channels, num_repetitions)
        self.out_shape = ModuleShape(in_features, out_channels, num_repetitions)

        # ========== 4. WEIGHT INITIALIZATION ==========
        # Linear sum: weight over input channels only (not cross-product)
        self.weights_shape = (
            self.out_shape.features,  # D_out
            self.out_shape.channels,  # O (output channels)
            self.out_shape.repetitions,  # R
            self._in_channels,  # C (input channels - linear, not cross-product)
        )

        if weights is None:
            # Initialize weights randomly, normalized over input channels
            weights = torch.rand(self.weights_shape) + 1e-08
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # Validate weights shape
        if weights.shape != self.weights_shape:
            raise ValueError(
                f"Weight shape mismatch: expected {self.weights_shape}, got {weights.shape}"
            )

        # Register logits parameter
        self.logits = nn.Parameter(torch.zeros(self.weights_shape))

        # Set weights via property (converts to logits)
        self.weights = weights

    @property
    def feature_to_scope(self) -> np.ndarray:
        """Mapping from output features to their scopes."""
        if self._two_inputs:
            # Combine scopes from left and right inputs
            left_scopes = self.inputs[0].feature_to_scope
            right_scopes = self.inputs[1].feature_to_scope
            combined = []
            for r in range(self.out_shape.repetitions):
                rep_scopes = []
                for f in range(self.out_shape.features):
                    left_s = left_scopes[f, r]
                    right_s = right_scopes[f, r]
                    rep_scopes.append(Scope.join_all([left_s, right_s]))
                combined.append(np.array(rep_scopes))
            return np.stack(combined, axis=1)
        else:
            # Single input split into halves - combine adjacent pairs
            input_scopes = self.inputs.inputs.feature_to_scope
            combined = []
            for r in range(self.out_shape.repetitions):
                rep_scopes = []
                for f in range(self.out_shape.features):
                    left_s = input_scopes[2 * f, r]
                    right_s = input_scopes[2 * f + 1, r]
                    rep_scopes.append(Scope.join_all([left_s, right_s]))
                combined.append(np.array(rep_scopes))
            return np.stack(combined, axis=1)

    @property
    def log_weights(self) -> Tensor:
        """Log-normalized weights (sum to 1 over input channels)."""
        return torch.nn.functional.log_softmax(self.logits, dim=-1)

    @property
    def weights(self) -> Tensor:
        """Normalized weights (sum to 1 over input channels)."""
        return torch.nn.functional.softmax(self.logits, dim=-1)

    @weights.setter
    def weights(self, values: Tensor) -> None:
        """Set weights (must be positive and sum to 1 over channels)."""
        if values.shape != self.weights_shape:
            raise ValueError(f"Weight shape mismatch: expected {self.weights_shape}, got {values.shape}")
        if not torch.all(values > 0):
            raise ValueError("Weights must be positive.")
        sums = values.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums)):
            raise ValueError("Weights must sum to 1 over input channels.")
        # Project to logits space
        self.logits.data = proj_convex_to_real(values)

    @log_weights.setter
    def log_weights(self, values: Tensor) -> None:
        """Set log weights directly."""
        if values.shape != self.weights_shape:
            raise ValueError(f"Log weight shape mismatch: expected {self.weights_shape}, got {values.shape}")
        self.logits.data = values

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, weights={self.weights_shape}"

    def _get_left_right_ll(
        self, data: Tensor, cache: Cache | None = None
    ) -> tuple[Tensor, Tensor]:
        """Get log-likelihoods from left and right children.

        Returns:
            Tuple of (left_ll, right_ll), each of shape (batch, features, channels, reps).
        """
        if self._two_inputs:
            left_ll = self.inputs[0].log_likelihood(data, cache=cache)
            right_ll = self.inputs[1].log_likelihood(data, cache=cache)
        else:
            # SplitConsecutive returns list of [left, right]
            lls = self.inputs.log_likelihood(data, cache=cache)
            left_ll = lls[0]
            right_ll = lls[1]
        return left_ll, right_ll

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log-likelihood using linear sum over channels.

        Unlike EinsumLayer which computes cross-product (I × J), this computes
        a linear combination: add left+right (product), then logsumexp over channels.

        Args:
            data: Input data of shape (batch_size, num_features).
            cache: Optional cache for intermediate results.

        Returns:
            Log-likelihood tensor of shape (batch, out_features, out_channels, reps).
        """
        # Get child log-likelihoods
        left_ll, right_ll = self._get_left_right_ll(data, cache)

        # Dimensions: N=batch, D=features, C=channels, R=reps
        N, D, C, R = left_ll.size()

        # Product: left + right in log-space
        # Shape: (N, D, C, R)
        prod_ll = left_ll + right_ll

        # Expand for output channels dimension
        # prod_ll: (N, D, C, R) -> (N, D, 1, C, R)
        prod_ll = prod_ll.unsqueeze(2)

        # Get log weights: (D, O, R, C) -> (1, D, O, C, R)
        log_weights = self.log_weights.permute(0, 1, 3, 2).unsqueeze(0)

        # Weighted sum over input channels
        # (N, D, 1, C, R) + (1, D, O, C, R) -> (N, D, O, C, R)
        weighted_ll = prod_ll + log_weights

        # LogSumExp over input channels (dim=3)
        log_prob = torch.logsumexp(weighted_ll, dim=3)  # (N, D, O, R)

        return log_prob

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Sample from the LinsumLayer.

        Args:
            num_samples: Number of samples to generate.
            data: Optional data tensor with evidence (NaN for missing).
            is_mpe: Whether to perform MPE instead of sampling.
            cache: Optional cache with log-likelihoods for conditional sampling.
            sampling_ctx: Sampling context with channel indices.

        Returns:
            Sampled data tensor.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        if cache is None:
            cache = Cache()

        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        # Get logits and select based on context
        logits = self.logits  # (D, O, R, C)

        # Expand for batch dimension
        batch_size = sampling_ctx.channel_index.shape[0]
        logits = logits.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        # logits shape: (B, D, O, R, C)

        # Select output channel based on parent's channel_index
        channel_idx = sampling_ctx.channel_index  # (B, D)

        # Gather the correct output channel
        idx = channel_idx.view(batch_size, self.out_shape.features, 1, 1, 1)
        idx = idx.expand(-1, -1, -1, self.out_shape.repetitions, self._in_channels)
        logits = logits.gather(dim=2, index=idx).squeeze(2)
        # logits shape: (B, D, R, C)

        # Select repetition if specified
        if sampling_ctx.repetition_idx is not None:
            rep_idx = sampling_ctx.repetition_idx.view(-1, 1, 1, 1)
            rep_idx = rep_idx.expand(-1, self.out_shape.features, -1, self._in_channels)
            logits = logits.gather(dim=2, index=rep_idx).squeeze(2)
            # logits shape: (B, D, C)
        else:
            if self.out_shape.repetitions > 1:
                raise ValueError(
                    "repetition_idx must be provided when sampling with num_repetitions > 1"
                )
            logits = logits[:, :, 0, :]  # (B, D, C)

        # Condition on evidence if cache has log-likelihoods
        if self._two_inputs:
            left_cache_key = self.inputs[0]
            right_cache_key = self.inputs[1]
        else:
            left_cache_key = "linsum_left"
            right_cache_key = "linsum_right"

        if (
            cache is not None
            and "log_likelihood" in cache
            and cache["log_likelihood"].get(left_cache_key) is not None
            and cache["log_likelihood"].get(right_cache_key) is not None
        ):
            # Get cached log-likelihoods
            left_ll = cache["log_likelihood"][left_cache_key]  # (B, D, C, R)
            right_ll = cache["log_likelihood"][right_cache_key]  # (B, D, C, R)

            # Select repetition
            if sampling_ctx.repetition_idx is not None:
                rep_idx = sampling_ctx.repetition_idx.view(-1, 1, 1, 1)
                rep_idx_l = rep_idx.expand(-1, left_ll.shape[1], left_ll.shape[2], -1)
                left_ll = left_ll.gather(dim=-1, index=rep_idx_l).squeeze(-1)
                right_ll = right_ll.gather(dim=-1, index=rep_idx_l).squeeze(-1)

            # Product log-likelihood for each channel
            prod_ll = left_ll + right_ll  # (B, D, C)

            # Compute posterior
            log_prior = logits
            log_posterior = log_prior + prod_ll
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=-1, keepdim=True)
            logits = log_posterior

        # Sample or MPE
        if is_mpe:
            indices = logits.argmax(dim=-1)  # (B, D)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            indices = dist.sample()  # (B, D)

        # Sample from left and right children with same channel index
        # (Linear combination means left and right use the same channel)
        if self._two_inputs:
            # Left child
            left_ctx = sampling_ctx.copy()
            left_ctx.channel_index = indices
            self.inputs[0].sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=left_ctx)

            # Right child
            right_ctx = sampling_ctx.copy()
            right_ctx.channel_index = indices
            self.inputs[1].sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=right_ctx)
        else:
            # Single input with Split module - use generic merge_split_indices
            # For LinsumLayer, both left and right use the same indices (linear combination)
            full_indices = self.inputs.merge_split_indices(indices, indices)
            full_mask = sampling_ctx.mask.repeat(1, 2)

            child_ctx = sampling_ctx.copy()
            child_ctx.update(channel_index=full_indices, mask=full_mask)
            self.inputs.sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=child_ctx)

        return data

    def expectation_maximization(
        self,
        data: Tensor,
        bias_correction: bool = True,
        cache: Cache | None = None,
    ) -> None:
        """Perform EM step to update weights.

        Args:
            data: Training data tensor.
            bias_correction: Whether to apply bias correction.
            cache: Cache with log-likelihoods.
        """
        if cache is None:
            cache = Cache()

        with torch.no_grad():
            # Get cached values
            left_ll, right_ll = self._get_left_right_ll(data, cache)

            module_lls = cache["log_likelihood"].get(self)
            if module_lls is None:
                raise ValueError("Module log-likelihoods not in cache. Call log_likelihood first.")

            # E-step: compute expected counts
            log_weights = self.log_weights.unsqueeze(0)  # (1, D, O, R, C)

            # Product of left and right
            prod_ll = left_ll + right_ll  # (B, D, C, R)
            # Rearrange to match weights: (B, D, O, R, C)
            prod_ll = prod_ll.permute(0, 1, 3, 2).unsqueeze(2)  # (B, D, 1, R, C)

            # Get gradients
            log_grads = torch.log(module_lls.grad + 1e-10)

            log_grads = log_grads.unsqueeze(-1)  # (B, D, O, R, 1)
            module_lls = module_lls.unsqueeze(-1)  # (B, D, O, R, 1)

            # Compute log expectations
            log_expectations = log_weights + log_grads + prod_ll - module_lls
            log_expectations = log_expectations.logsumexp(0)  # Sum over batch

            # Normalize to get new log weights
            new_log_weights = torch.nn.functional.log_softmax(log_expectations, dim=-1)

            # M-step: update weights
            self.log_weights = new_log_weights

        # Recurse to children
        if self._two_inputs:
            self.inputs[0].expectation_maximization(data, bias_correction=bias_correction, cache=cache)
            self.inputs[1].expectation_maximization(data, bias_correction=bias_correction, cache=cache)
        else:
            self.inputs.inputs.expectation_maximization(data, bias_correction=bias_correction, cache=cache)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        bias_correction: bool = True,
        nan_strategy: str = "ignore",
        cache: Cache | None = None,
    ) -> None:
        """MLE step (equivalent to EM for sum nodes)."""
        self.expectation_maximization(data, bias_correction=bias_correction, cache=cache)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Optional["LinsumLayer" | Module]:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: Random variable indices to marginalize.
            prune: Whether to prune unnecessary modules.
            cache: Cache for memoization.

        Returns:
            Marginalized module or None if fully marginalized.
        """
        if cache is None:
            cache = Cache()

        module_scope = self.scope
        mutual_rvs = set(module_scope.query).intersection(set(marg_rvs))

        # Fully marginalized
        if len(mutual_rvs) == len(module_scope.query):
            return None

        # No overlap - return self unchanged
        if not mutual_rvs:
            return self

        # Partially marginalized
        if self._two_inputs:
            left_marg = self.inputs[0].marginalize(marg_rvs, prune=prune, cache=cache)
            right_marg = self.inputs[1].marginalize(marg_rvs, prune=prune, cache=cache)

            if left_marg is None and right_marg is None:
                return None
            elif left_marg is None:
                return right_marg
            elif right_marg is None:
                return left_marg
            else:
                # Both still exist - create new LinsumLayer with marginalized children
                return LinsumLayer(
                    inputs=[left_marg, right_marg],
                    out_channels=self.out_shape.channels,
                    num_repetitions=self.out_shape.repetitions,
                )
        else:
            # Single input - marginalize the underlying input
            marg_input = self.inputs.inputs.marginalize(marg_rvs, prune=prune, cache=cache)
            if marg_input is None:
                return None
            # Check if we still have enough features for LinsumLayer
            if marg_input.out_shape.features < 2:
                return marg_input
            if marg_input.out_shape.features % 2 != 0:
                # Odd number of features - can't use LinsumLayer
                return marg_input
            return LinsumLayer(
                inputs=marg_input,
                out_channels=self.out_shape.channels,
                num_repetitions=self.out_shape.repetitions,
            )
