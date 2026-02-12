"""EinsumLayer for efficient sum-product operations in probabilistic circuits.

Implements the EinsumLayer as described in the Einet paper, combining product
and sum operations into a single efficient einsum operation using the
LogEinsumExp trick for numerical stability.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from spflow.exceptions import InvalidWeightsError, MissingCacheError, ScopeError, ShapeError
from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.split import Split, SplitMode
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.utils.cache import Cache, cached
from spflow.utils.projections import proj_convex_to_real
from spflow.utils.sampling_context import SamplingContext, require_sampling_context


class EinsumLayer(Module):
    """EinsumLayer combining product and sum operations efficiently.

    Implements sum(product(x)) using einsum for circuits with arbitrary tree
    structure. Takes pairs of adjacent features as left/right children, computes
    their cross-product over channels, and sums with learned weights.

    The LogEinsumExp trick is used for numerical stability in log-space.

    Attributes:
        logits (Parameter): Unnormalized log-weights for gradient optimization.
        unraveled_channel_indices (Tensor): Mapping from flat to (i,j) channel pairs.
    """

    def __init__(
        self,
        inputs: Module | list[Module],
        out_channels: int,
        num_repetitions: int | None = None,
        weights: Tensor | None = None,
        split_mode: SplitMode | None = None,
    ) -> None:
        """Initialize EinsumLayer.

        Args:
            inputs: Either a single module (features will be split into pairs)
                or a list of exactly two modules (left and right children).
            out_channels: Number of output sum nodes per feature.
            num_repetitions: Number of repetitions. If None, inferred from inputs.
            weights: Optional initial weights tensor. If provided, must have shape
                (out_features, out_channels, num_repetitions, left_channels, right_channels).
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
                    f"EinsumLayer requires exactly 2 input modules when given a list, got {len(inputs)}."
                )
            self._two_inputs = True
            left_input, right_input = inputs
            # Validate compatible shapes (channels can differ for cross-product)
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
                raise ScopeError("Left and right input scopes must be disjoint.")

            self.inputs = nn.ModuleList([left_input, right_input])
            self._left_channels = left_input.out_shape.channels
            self._right_channels = right_input.out_shape.channels
            in_features = left_input.out_shape.features
            if num_repetitions is None:
                num_repetitions = left_input.out_shape.repetitions
            self.scope = Scope.join_all([left_input.scope, right_input.scope])

        else:
            # Single input: will split features into left/right halves
            self._two_inputs = False
            if inputs.out_shape.features < 2:
                raise ValueError(
                    f"EinsumLayer requires at least 2 input features for splitting, "
                    f"got {inputs.out_shape.features}."
                )
            if inputs.out_shape.features % 2 != 0:
                raise ValueError(
                    f"EinsumLayer requires even number of input features for splitting, "
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
            self._left_channels = inputs.out_shape.channels
            self._right_channels = inputs.out_shape.channels
            in_features = inputs.out_shape.features // 2
            if num_repetitions is None:
                num_repetitions = inputs.out_shape.repetitions
            self.scope = inputs.scope

        # ========== 2. CONFIGURATION VALIDATION ==========
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}.")

        # ========== 3. SHAPE COMPUTATION ==========
        # Use max channels for in_shape (for informational purposes)
        max_in_channels = max(self._left_channels, self._right_channels)
        self.in_shape = ModuleShape(in_features, max_in_channels, num_repetitions)
        self.out_shape = ModuleShape(in_features, out_channels, num_repetitions)

        # ========== 4. WEIGHT INITIALIZATION ==========
        self.weights_shape = (
            self.out_shape.features,  # D_out
            self.out_shape.channels,  # O (output channels)
            self.out_shape.repetitions,  # R
            self._left_channels,  # I (left input channels)
            self._right_channels,  # J (right input channels)
        )

        # Create index mapping for sampling: flatten (i,j) -> idx and back
        self.register_buffer(
            "unraveled_channel_indices",
            torch.tensor(
                [(i, j) for i in range(self._left_channels) for j in range(self._right_channels)],
                dtype=torch.long,
            ),
        )

        if weights is None:
            # Initialize weights randomly, normalized over (i,j) pairs
            weights = torch.rand(self.weights_shape) + 1e-08
            weights = weights / weights.sum(dim=(-2, -1), keepdim=True)

        # Validate weights shape
        if weights.shape != self.weights_shape:
            raise ValueError(f"Weight shape mismatch: expected {self.weights_shape}, got {weights.shape}")

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
        """Log-normalized weights (sum to 1 over input channel pairs)."""
        # Flatten input-channel pair axes before normalization.
        flat_logits = rearrange(self.logits, "f co r i j -> f co r (i j)")
        log_weights = torch.nn.functional.log_softmax(flat_logits, dim=-1)
        return rearrange(
            log_weights,
            "f co r (i j) -> f co r i j",
            i=self._left_channels,
            j=self._right_channels,
        )

    @property
    def weights(self) -> Tensor:
        """Normalized weights (sum to 1 over input channel pairs)."""
        flat_logits = rearrange(self.logits, "f co r i j -> f co r (i j)")
        weights = torch.nn.functional.softmax(flat_logits, dim=-1)
        return rearrange(
            weights,
            "f co r (i j) -> f co r i j",
            i=self._left_channels,
            j=self._right_channels,
        )

    @weights.setter
    def weights(self, values: Tensor) -> None:
        """Set weights (must be positive and sum to 1 over i,j pairs)."""
        if values.shape != self.weights_shape:
            raise ShapeError(f"Weight shape mismatch: expected {self.weights_shape}, got {values.shape}")
        if not torch.all(values > 0):
            raise InvalidWeightsError("Weights must be positive.")
        sums = values.sum(dim=(-2, -1))
        if not torch.allclose(sums, torch.ones_like(sums)):
            raise InvalidWeightsError("Weights must sum to 1 over (i,j) channel pairs.")
        # Project to logits space
        flat_weights = rearrange(values, "f co r i j -> f co r (i j)")
        flat_logits = proj_convex_to_real(flat_weights)
        self.logits.data = rearrange(
            flat_logits,
            "f co r (i j) -> f co r i j",
            i=self._left_channels,
            j=self._right_channels,
        )

    @log_weights.setter
    def log_weights(self, values: Tensor) -> None:
        """Set log weights directly."""
        if values.shape != self.weights_shape:
            raise ShapeError(f"Log weight shape mismatch: expected {self.weights_shape}, got {values.shape}")
        self.logits.data = values

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, weights={self.weights_shape}"

    def _get_left_right_ll(self, data: Tensor, cache: Cache | None = None) -> tuple[Tensor, Tensor]:
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
        """Compute log-likelihood using LogEinsumExp trick.

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

        # LogEinsumExp trick for numerical stability
        # Compute max for normalization
        left_max = torch.max(left_ll, dim=2, keepdim=True)[0]  # (N, D, 1, R)
        left_prob = torch.exp(left_ll - left_max)  # (N, D, C, R)

        right_max = torch.max(right_ll, dim=2, keepdim=True)[0]  # (N, D, 1, R)
        right_prob = torch.exp(right_ll - right_max)  # (N, D, C, R)

        # Get normalized weights
        weights = self.weights  # (D, O, R, I, J)

        # Einsum: product over channels, weighted sum
        # n=batch, d=features, i=left_channels, j=right_channels, o=out_channels, r=reps
        prob = torch.einsum("ndir,ndjr,dorij->ndor", left_prob, right_prob, weights)

        # Re-add the log maxes
        log_prob = torch.log(prob) + left_max + right_max

        return log_prob

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Sample from the EinsumLayer.

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

        sampling_ctx = require_sampling_context(
            sampling_ctx,
            module_name=self.__class__.__name__,
            num_samples=data.shape[0],
            module_out_shape=self.out_shape,
            device=data.device,
        )

        # Get logits and select based on context
        logits = self.logits  # (D, O, R, I, J)

        # Expand for batch dimension
        batch_size = int(sampling_ctx.channel_index.shape[0])
        logits = repeat(logits, "f co r i j -> b f co r i j", b=batch_size)
        # logits shape: (B, D, O, R, I, J)

        # Select output channel based on parent's channel_index
        # sampling_ctx.channel_index: (B, D) - indices into out_channels
        channel_idx = sampling_ctx.channel_index  # (B, D)

        # Gather the correct output channel
        # Expand channel_idx to match logits dimensions
        num_repetitions = self.out_shape.repetitions
        num_left_channels = self._left_channels
        num_right_channels = self._right_channels
        idx = repeat(
            channel_idx,
            "b f -> b f 1 r i j",
            r=num_repetitions,
            i=num_left_channels,
            j=num_right_channels,
        )
        logits = logits.gather(dim=2, index=idx)
        logits = rearrange(logits, "b f 1 r i j -> b f r i j")
        # logits shape: (B, D, R, I, J)

        # Select repetition if specified
        if sampling_ctx.repetition_idx is not None:
            num_features = self.out_shape.features
            rep_idx = repeat(
                rearrange(sampling_ctx.repetition_idx, "... -> (...)"),
                "b -> b f 1 i j",
                f=num_features,
                i=num_left_channels,
                j=num_right_channels,
            )
            logits = logits.gather(dim=2, index=rep_idx)
            logits = rearrange(logits, "b f 1 i j -> b f i j")
            # logits shape: (B, D, I, J)
        else:
            if self.out_shape.repetitions > 1:
                raise ValueError("repetition_idx must be provided when sampling with num_repetitions > 1")
            logits = logits[:, :, 0, :, :]

        # Flatten (I, J) for categorical sampling
        logits_flat = rearrange(logits, "b f i j -> b f (i j)")

        # Condition on evidence if cache has log-likelihoods.
        left_ll = None
        right_ll = None
        if cache is not None and "log_likelihood" in cache:
            if self._two_inputs:
                left_ll = cache["log_likelihood"].get(self.inputs[0])
                right_ll = cache["log_likelihood"].get(self.inputs[1])
            else:
                split_ll = cache["log_likelihood"].get(self.inputs)
                if isinstance(split_ll, (list, tuple)) and len(split_ll) == 2:
                    left_ll, right_ll = split_ll

        if left_ll is not None and right_ll is not None:
            # Select repetition
            if sampling_ctx.repetition_idx is not None:
                num_features = int(left_ll.shape[1])
                num_left_channels = int(left_ll.shape[2])
                rep_idx_l = repeat(
                    rearrange(sampling_ctx.repetition_idx, "... -> (...)"),
                    "b -> b f i 1",
                    f=num_features,
                    i=num_left_channels,
                )
                left_ll = left_ll.gather(dim=-1, index=rep_idx_l)
                right_ll = right_ll.gather(dim=-1, index=rep_idx_l)
                left_ll = rearrange(left_ll, "b f i 1 -> b f i")
                right_ll = rearrange(right_ll, "b f j 1 -> b f j")

            # Compute joint log-likelihood for each (i, j) pair
            # left_ll: (B, D, I), right_ll: (B, D, J)
            left_ll = rearrange(left_ll, "b f i -> b f i 1")
            right_ll = rearrange(right_ll, "b f j -> b f 1 j")
            joint_ll = left_ll + right_ll  # (B, D, I, J)
            joint_ll_flat = rearrange(joint_ll, "b f i j -> b f (i j)")

            # Compute posterior
            log_prior = logits_flat
            log_posterior = log_prior + joint_ll_flat
            log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=-1, keepdim=True)
            logits_flat = log_posterior

        # Sample or MPE
        if is_mpe:
            indices = logits_flat.argmax(dim=-1)  # (B, D)
        else:
            dist = torch.distributions.Categorical(logits=logits_flat)
            indices = dist.sample()  # (B, D)

        # Unravel indices to (i, j) pairs
        ij_indices = self.unraveled_channel_indices[indices]  # (B, D, 2)
        left_indices = ij_indices[..., 0]  # (B, D)
        right_indices = ij_indices[..., 1]  # (B, D)

        # Sample from left and right children
        if self._two_inputs:
            # Left child
            left_ctx = sampling_ctx.copy()
            left_ctx.channel_index = left_indices
            self.inputs[0].sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=left_ctx)

            # Right child
            right_ctx = sampling_ctx.copy()
            right_ctx.channel_index = right_indices
            self.inputs[1].sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=right_ctx)
        else:
            # Single input with Split module - use generic merge_split_indices
            full_indices = self.inputs.merge_split_indices(left_indices, right_indices)
            full_mask = repeat(sampling_ctx.mask, "b f -> b (f two)", two=2)

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
                raise MissingCacheError("Module log-likelihoods not in cache. Call log_likelihood first.")

            # E-step: compute expected counts
            log_weights = rearrange(self.log_weights, "f co r i j -> 1 f co r i j")

            left_ll = rearrange(left_ll, "b f i r -> b f 1 r i 1")
            right_ll = rearrange(right_ll, "b f j r -> b f 1 r 1 j")

            # Get gradients (how much each output contributed)
            log_grads = torch.log(module_lls.grad + 1e-10)

            log_grads = rearrange(log_grads, "b f co r -> b f co r 1 1")
            module_lls = rearrange(module_lls, "b f co r -> b f co r 1 1")

            # Joint input log-likelihood
            joint_input_ll = left_ll + right_ll  # (B, D, 1, R, I, J)

            # Compute log expectations
            log_expectations = log_weights + log_grads + joint_input_ll - module_lls
            log_expectations = log_expectations.logsumexp(0)  # Sum over batch

            # Normalize to get new log weights
            flat_expectations = rearrange(log_expectations, "f co r i j -> f co r (i j)")
            flat_log_weights = torch.nn.functional.log_softmax(flat_expectations, dim=-1)
            new_log_weights = rearrange(
                flat_log_weights,
                "f co r (i j) -> f co r i j",
                i=self._left_channels,
                j=self._right_channels,
            )

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
    ) -> Optional["EinsumLayer" | Module]:
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

        # No overlap - return self unchanged (inputs unchanged)
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
                # Both still exist - create new EinsumLayer with marginalized children
                return EinsumLayer(
                    inputs=[left_marg, right_marg],
                    out_channels=self.out_shape.channels,
                    num_repetitions=self.out_shape.repetitions,
                )
        else:
            # Single input - marginalize the underlying input
            marg_input = self.inputs.inputs.marginalize(marg_rvs, prune=prune, cache=cache)
            if marg_input is None:
                return None
            # Check if we still have enough features for EinsumLayer
            if marg_input.out_shape.features < 2:
                return marg_input
            if marg_input.out_shape.features % 2 != 0:
                # Odd number of features - can't use EinsumLayer
                return marg_input
            return EinsumLayer(
                inputs=marg_input,
                out_channels=self.out_shape.channels,
                num_repetitions=self.out_shape.repetitions,
            )
