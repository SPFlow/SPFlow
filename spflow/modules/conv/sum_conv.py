"""Convolutional sum layer for probabilistic circuits.

Provides SumConv, which applies learned weighted sums over input channels
within spatial patches, enabling mixture modeling with spatial structure.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import functional as F

from spflow.exceptions import InvalidWeightsError, MissingCacheError, ShapeError
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.cache import Cache, cached
from spflow.utils.projections import proj_convex_to_real
from spflow.utils.sampling_context import SamplingContext, validate_sampling_context


class SumConv(Module):
    """Convolutional sum layer for probabilistic circuits.

    Applies weighted sum over input channels within spatial patches. Weights are
    learned and normalized to sum to one per patch position, maintaining valid
    probability distributions. Useful for modeling spatial structure in image data.

    The layer expects input with spatial structure and applies shared weights
    across all spatial patches of the same position within the kernel.

    Attributes:
        inputs (Module): Input module providing log-likelihoods.
        kernel_size (int): Size of the spatial kernel (kernel_size x kernel_size).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (mixture components).
        logits (Parameter): Unnormalized log-weights for gradient optimization.
    """

    def __init__(
        self,
        inputs: Module,
        out_channels: int,
        kernel_size: int,
        num_repetitions: int = 1,
    ) -> None:
        """Create a SumConv module for spatial mixture modeling.

        Args:
            inputs: Input module providing log-likelihoods with spatial structure.
            out_channels: Number of output mixture components.
            kernel_size: Size of the spatial kernel (kernel_size x kernel_size).
            num_repetitions: Number of independent repetitions.

        Raises:
            ValueError: If out_channels < 1 or kernel_size < 1.
        """
        super().__init__()

        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

        self.inputs = inputs
        self.kernel_size = kernel_size
        self.sum_dim = 1  # Sum over input channels

        # Infer input shape
        input_shape = self.inputs.out_shape
        self.in_channels = input_shape.channels

        # Scope is inherited from input (per-pixel scopes preserved)
        self.scope = self.inputs.scope

        # Shape computation
        self.in_shape = input_shape
        self.out_shape = ModuleShape(
            features=input_shape.features,  # Spatial dimensions unchanged
            channels=out_channels,
            repetitions=num_repetitions,
        )

        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size, repetitions)
        self.weights_shape = (
            out_channels,
            self.in_channels,
            kernel_size,
            kernel_size,
            num_repetitions,
        )

        # Initialize weights uniformly
        weights = torch.rand(self.weights_shape) + 1e-08
        weights = weights / weights.sum(dim=self.sum_dim, keepdim=True)

        # Register parameter for unnormalized log-probabilities
        self.logits = torch.nn.Parameter(proj_convex_to_real(weights))

    @property
    def feature_to_scope(self) -> np.ndarray:
        """Per-pixel scopes are preserved from input."""
        return self.inputs.feature_to_scope

    @property
    def log_weights(self) -> Tensor:
        """Returns the log weights normalized to sum to one over input channels.

        Returns:
            Tensor: Log weights of shape (out_c, in_c, k, k, reps).
        """
        return F.log_softmax(self.logits, dim=self.sum_dim)

    @property
    def weights(self) -> Tensor:
        """Returns the weights normalized to sum to one over input channels.

        Returns:
            Tensor: Weights of shape (out_c, in_c, k, k, reps).
        """
        return F.softmax(self.logits, dim=self.sum_dim)

    @weights.setter
    def weights(self, values: Tensor) -> None:
        """Set weights of all nodes.

        Args:
            values: Tensor containing weights.

        Raises:
            ValueError: If weights have invalid shape or values.
        """
        if values.shape != self.weights_shape:
            raise ShapeError(
                f"Invalid shape for weights: Was {values.shape} but expected {self.weights_shape}."
            )
        if not torch.all(values > 0):
            raise InvalidWeightsError("Weights must be all positive.")
        if not torch.allclose(values.sum(dim=self.sum_dim), values.new_tensor(1.0)):
            raise InvalidWeightsError("Weights must sum to one over input channels.")
        self.logits.data = proj_convex_to_real(values)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_shape.channels}, "
            f"kernel_size={self.kernel_size}"
        )

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood using convolutional weighted sum.

        Applies weighted sum over input channels within spatial patches.
        Each kernel position gets its own set of mixture weights.
        Uses logsumexp for numerical stability.

        Args:
            data: Input data of shape (batch_size, num_features).
            cache: Cache for intermediate computations.

        Returns:
            Tensor: Log-likelihood of shape (batch, features, out_channels, reps).
        """
        # Get input log-likelihoods: (batch, features, in_channels, reps)
        ll = self.inputs.log_likelihood(data, cache=cache)

        batch_size = ll.shape[0]
        num_features = ll.shape[1]
        in_channels = ll.shape[2]
        in_reps = ll.shape[3]

        # Handle repetition matching
        out_reps = self.out_shape.repetitions
        if in_reps == 1 and out_reps > 1:
            raise RuntimeError(
                "Input repetitions cannot be broadcast to multiple output repetitions in SumConv.log_likelihood."
            )
        elif in_reps != out_reps and in_reps != 1:
            raise ValueError(f"Input repetitions {in_reps} incompatible with output {out_reps}")

        # Infer spatial dimensions from num_features
        # Assume square spatial dimensions
        H = W = int(num_features**0.5)
        if H * W != num_features:
            raise ValueError(
                f"SumConv requires square spatial dimensions. Got {num_features} features "
                f"which is not a perfect square."
            )

        K = self.kernel_size

        # Special case: spatial dims smaller than kernel size
        # Use only the first kernel weight position [0, 0]
        if H < K or W < K:
            # Get log weights for position [0, 0]: (out_c, in_c, reps)
            log_weights = self.log_weights[:, :, 0, 0, :]  # (out_c, in_c, reps)

            # Reshape for broadcasting: (1, 1, out_c, in_c, reps)
            log_weights = rearrange(log_weights, "co ci r -> 1 1 co ci r")

            # Reshape ll for broadcasting: (batch, features, 1, in_c, reps)
            ll = rearrange(ll, "b f ci r -> b f 1 ci r")

            # Weighted sum over input channels: logsumexp over dim 3 (in_channels)
            weighted_lls = ll + log_weights
            result = torch.logsumexp(weighted_lls, dim=3)  # (batch, features, out_c, reps)

            return result

        if H % K != 0 or W % K != 0:
            raise ValueError(f"Spatial dims ({H}, {W}) must be divisible by kernel_size {K}")

        # Get log weights: (out_c, in_c, k, k, reps)
        log_weights = self.log_weights

        # Reshape ll from (batch, features, in_c, reps) to spatial form:
        # (batch, in_c, H, W, reps)
        ll = rearrange(ll, "b (h w) ci r -> b ci h w r", h=H, w=W)

        # Patch the input into KxK blocks
        # (batch, in_c, H//K, K, W//K, K, reps)
        ll = rearrange(ll, "b ci (oh kh) (ow kw) r -> b ci oh ow kh kw r", kh=K, kw=K)

        # Make space for out_channels: (batch, 1, in_c, H//K, W//K, K, K, reps)
        ll = rearrange(ll, "b ci oh ow kh kw r -> b 1 ci oh ow kh kw r")

        # Make space in log_weights for spatial dims: (1, out_c, in_c, 1, 1, K, K, reps)
        log_weights = rearrange(log_weights, "co ci kh kw r -> 1 co ci 1 1 kh kw r")

        # Weighted sum over input channels: logsumexp over dim 2 (in_channels)
        weighted_lls = ll + log_weights
        result = torch.logsumexp(weighted_lls, dim=2)  # (batch, out_c, H//K, W//K, K, K, reps)

        # Invert the patch transformation and flatten spatial dimensions.
        result = rearrange(result, "b co oh ow kh kw r -> b (oh kh ow kw) co r")

        return result

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        """Generate samples from sum conv module.

        Each spatial position samples from its per-position kernel weights.

        Args:
            num_samples: Number of samples to generate.
            data: Data tensor with NaN values to fill with samples.
            is_mpe: Whether to perform maximum a posteriori estimation.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Tensor: Sampled values.
        """
        batch_size = data.shape[0]

        num_features = self.in_shape.features

        # Infer spatial dimensions
        H = W = int(num_features**0.5)
        K = self.kernel_size

        if H * W != num_features:
            raise ValueError(
                f"SumConv requires square spatial dimensions. Got {num_features} features "
                f"which is not a perfect square."
            )

        if H % K != 0 or W % K != 0:
            raise ValueError(f"Spatial dims ({H}, {W}) must be divisible by kernel_size {K}")

        validate_sampling_context(
            sampling_ctx,
            num_samples=data.shape[0],
            num_features=self.out_shape.features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, num_features),
        )
        sampling_ctx.broadcast_feature_width(target_features=num_features, allow_from_one=True)

        channel_idx = sampling_ctx.channel_index  # (batch, H*W)

        # Get logits: (out_c, in_c, k, k, reps)
        logits = self.logits

        # Select repetition
        if sampling_ctx.repetition_index is not None:
            # logits: (out_c, in_c, k, k, reps) -> select reps
            out_channels = int(logits.shape[0])
            in_channels = int(logits.shape[1])
            rep_idx = repeat(
                rearrange(sampling_ctx.repetition_index, "... -> (...)"),
                "b -> b co ci kh kw 1",
                co=out_channels,
                ci=in_channels,
                kh=K,
                kw=K,
            )
            logits = repeat(logits, "co ci kh kw r -> b co ci kh kw r", b=batch_size)
            logits = torch.gather(logits, dim=-1, index=rep_idx)
            logits = rearrange(logits, "b co ci kh kw 1 -> b co ci kh kw")
            # logits: (batch, out_c, in_c, k, k)
        else:
            logits = logits[..., 0]  # (out_c, in_c, k, k)
            logits = repeat(logits, "co ci kh kw -> b co ci kh kw", b=batch_size)
            # logits: (batch, out_c, in_c, k, k)

        # Check for cached likelihoods (conditional sampling)
        input_lls = None
        if "log_likelihood" in cache and cache["log_likelihood"].get(self.inputs) is not None:
            input_lls = cache["log_likelihood"][self.inputs]  # (batch, features, in_c, reps)

            # Select repetition
            if sampling_ctx.repetition_index is not None:
                num_features = int(input_lls.shape[1])
                num_input_channels = int(input_lls.shape[2])
                rep_idx = repeat(
                    rearrange(sampling_ctx.repetition_index, "... -> (...)"),
                    "b -> b f ci 1",
                    f=num_features,
                    ci=num_input_channels,
                )
                input_lls = torch.gather(input_lls, dim=-1, index=rep_idx)
                input_lls = rearrange(input_lls, "b f ci 1 -> b f ci")
            else:
                input_lls = input_lls[..., 0]
            # input_lls: (batch, H*W, in_c)

            # Reshape to spatial: (batch, H, W, in_c)
            input_lls = rearrange(input_lls, "b (h w) ci -> b h w ci", h=H, w=W)

        # Reshape channel_idx to spatial: (batch, H, W)
        channel_idx = rearrange(channel_idx, "b (h w) -> b h w", h=H, w=W)

        # Sample per-position: each pixel position needs its own sample
        # Create position indices for kernel
        # Position within kernel: pixel (i, j) has kernel pos (i % K, j % K)
        row_pos = repeat(torch.arange(H, device=data.device), "h -> b h w", b=batch_size, w=W)
        col_pos = repeat(torch.arange(W, device=data.device), "w -> b h w", b=batch_size, h=H)
        k_row = row_pos % K  # (batch, H, W)
        k_col = col_pos % K  # (batch, H, W)

        # logits: (batch, out_c, in_c, k, k)
        # Select logits for each position based on parent channel and kernel position
        # First gather by parent channel: (batch, H, W, in_c, k, k)
        logits_per_pos = rearrange(logits, "b co ci kh kw -> b (kh kw) co ci")

        # Compute flat kernel index
        k_flat = k_row * K + k_col  # (batch, H, W)

        # Expand logits to per-pixel layout before kernel-index gathering.
        logits_per_pos_exp = repeat(logits_per_pos, "b kk co ci -> b kk h w co ci", h=H, w=W)
        # (batch, K*K, H, W, out_c, in_c) -> swap dims for gather
        logits_per_pos_exp = rearrange(logits_per_pos_exp, "b kk h w co ci -> b h w kk co ci")

        # Gather by k_flat
        num_output_channels = self.out_shape.channels
        num_input_channels = self.in_channels
        k_flat_exp2 = repeat(k_flat, "b h w -> b h w 1 co ci", co=num_output_channels, ci=num_input_channels)
        selected_logits = torch.gather(logits_per_pos_exp, dim=3, index=k_flat_exp2)
        selected_logits = rearrange(selected_logits, "b h w 1 co ci -> b h w co ci")
        # selected_logits: (batch, H, W, out_c, in_c)

        # Now select by parent channel: channel_idx (batch, H, W)
        parent_ch = repeat(channel_idx, "b h w -> b h w 1 ci", ci=num_input_channels)
        selected_logits = torch.gather(selected_logits, dim=3, index=parent_ch)
        selected_logits = rearrange(selected_logits, "b h w 1 ci -> b h w ci")
        # selected_logits: (batch, H, W, in_c)

        # Compute posterior if we have cached likelihoods
        if input_lls is not None:
            # input_lls: (batch, H, W, in_c)
            log_posterior = selected_logits + input_lls
            log_posterior = F.log_softmax(log_posterior, dim=-1)
        else:
            log_posterior = F.log_softmax(selected_logits, dim=-1)

        # Sample for each position
        log_posterior_flat = rearrange(log_posterior, "b h w ci -> (b h w) ci")
        if sampling_ctx.is_mpe:
            sampled_channels_flat = torch.argmax(log_posterior_flat, dim=-1)
        else:
            sampled_channels_flat = torch.distributions.Categorical(logits=log_posterior_flat).sample()

        sampled_channels = rearrange(sampled_channels_flat, "(b h w) -> b (h w)", b=batch_size, h=H, w=W)

        # Update sampling context
        sampling_ctx.channel_index = sampled_channels

        # Sample from input
        self.inputs._sample(
            data=data,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

        return data

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        """Perform expectation-maximization step to update weights.

        Follows the standard EM update pattern for sum nodes:
        1. Get cached log-likelihoods for input and this module
        2. Compute expectations using: log_weights + log_grads + input_lls - module_lls
        3. Normalize to get new log_weights

        Args:
            data: Input data tensor.
            bias_correction: Whether to apply bias correction (unused currently).
            cache: Cache dictionary with log-likelihoods from forward pass.

        Raises:
            MissingCacheError: If required log-likelihoods are not found in cache.
        """
        with torch.no_grad():
            # Get cached log-likelihoods
            input_lls = cache["log_likelihood"].get(self.inputs)
            if input_lls is None:
                raise MissingCacheError(
                    "Input log-likelihoods not found in cache. Call log_likelihood first."
                )

            module_lls = cache["log_likelihood"].get(self)
            if module_lls is None:
                raise MissingCacheError(
                    "Module log-likelihoods not found in cache. Call log_likelihood first."
                )

            # input_lls shape: (batch, features, in_channels, reps)
            # module_lls shape: (batch, features, out_channels, reps)
            # log_weights shape: (out_channels, in_channels, k, k, reps)

            # Get log gradients from module output.
            # Accessing `.grad` on non-leaf tensors without `retain_grad()` emits
            # a PyTorch warning, so only read it when it is expected to exist.
            grad_expected = module_lls.is_leaf or module_lls.retains_grad
            module_lls_grad = module_lls.grad if grad_expected else None

            if module_lls_grad is None and grad_expected:
                raise RuntimeError(
                    "Expected gradient for cached module log-likelihood, but found None. "
                    "This usually indicates a disconnected/non-differentiable computation path."
                )

            if module_lls_grad is None:
                # If no gradient, use uniform (this happens at the root)
                log_grads = torch.zeros_like(module_lls)
            else:
                log_grads = torch.log(module_lls_grad + 1e-10)

            # Current log weights: (out_c, in_c, k, k, reps)
            # Average over kernel spatial dims for simplicity
            log_weights = self.log_weights.mean(dim=(2, 3))  # (out_c, in_c, reps)

            # Reshape for broadcasting:
            # log_weights: (1, 1, out_c, in_c, reps)
            # log_grads: (batch, features, out_c, 1, reps)
            # input_lls: (batch, features, 1, in_c, reps)
            # module_lls: (batch, features, out_c, 1, reps)

            log_weights = rearrange(log_weights, "co ci r -> 1 1 co ci r")
            log_grads = rearrange(log_grads, "b f co r -> b f co 1 r")
            input_lls = rearrange(input_lls, "b f ci r -> b f 1 ci r")
            module_lls = rearrange(module_lls, "b f co r -> b f co 1 r")

            # Compute log expectations
            # This follows the standard EM derivation for mixture models
            log_expectations = log_weights + log_grads + input_lls - module_lls
            # Shape: (batch, features, out_c, in_c, reps)

            # Sum over batch and features dimensions
            log_expectations = torch.logsumexp(log_expectations, dim=0)  # (features, out_c, in_c, reps)
            log_expectations = torch.logsumexp(log_expectations, dim=0)  # (out_c, in_c, reps)

            # Normalize over in_channels (sum dimension for this module)
            # The sum_dim for SumConv is dimension 1 (in_channels)
            log_expectations = torch.log_softmax(log_expectations, dim=1)

            # Update log_weights: need to expand to full kernel shape
            # Current shape: (out_c, in_c, reps)
            # Target shape: (out_c, in_c, k, k, reps)
            k = self.kernel_size
            new_log_weights = rearrange(log_expectations, "co ci r -> co ci 1 1 r")
            new_log_weights = repeat(new_log_weights, "co ci 1 1 r -> co ci kh kw r", kh=k, kw=k)

            # Set new weights
            self.logits.data = new_log_weights.contiguous()

        # Recursively call EM on inputs
        self.inputs._expectation_maximization_step(data, cache=cache, bias_correction=bias_correction)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> SumConv | Module | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variable indices to marginalize.
            prune: Whether to prune unnecessary nodes.
            cache: Optional cache for storing intermediate results.

        Returns:
            SumConv | Module | None: Marginalized module or None if fully marginalized.
        """
        # Compute scope intersection
        layer_scope = self.scope
        mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

        # Fully marginalized
        if len(mutual_rvs) == len(layer_scope.query):
            return None

        # Marginalize input
        marg_input = self.inputs.marginalize(marg_rvs, prune=prune, cache=cache)
        if marg_input is None:
            return None

        # For now, return a new SumConv with marginalized input
        # Note: This is a simplified implementation
        return SumConv(
            inputs=marg_input,
            out_channels=self.out_shape.channels,
            kernel_size=self.kernel_size,
            num_repetitions=self.out_shape.repetitions,
        )
