"""Convolutional sum layer for probabilistic circuits.

Provides SumConv, which applies learned weighted sums over input channels
within spatial patches, enabling mixture modeling with spatial structure.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.cache import Cache, cached
from spflow.utils.projections import proj_convex_to_real
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context
from spflow.modules.conv.utils import expand_sampling_context, upsample_sampling_context


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
            raise ValueError(
                f"Invalid shape for weights: Was {values.shape} but expected {self.weights_shape}."
            )
        if not torch.all(values > 0):
            raise ValueError("Weights must be all positive.")
        if not torch.allclose(values.sum(dim=self.sum_dim), torch.tensor(1.0)):
            raise ValueError("Weights must sum to one over input channels.")
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
        if cache is None:
            cache = Cache()

        # Get input log-likelihoods: (batch, features, in_channels, reps)
        ll = self.inputs.log_likelihood(data, cache=cache)

        batch_size = ll.shape[0]
        num_features = ll.shape[1]
        in_channels = ll.shape[2]
        in_reps = ll.shape[3]

        # Handle repetition matching
        out_reps = self.out_shape.repetitions
        if in_reps == 1 and out_reps > 1:
            # Broadcast input reps
            ll = ll.unsqueeze(-1).expand(-1, -1, -1, out_reps)
        elif in_reps != out_reps and in_reps != 1:
            raise ValueError(f"Input repetitions {in_reps} incompatible with output {out_reps}")

        # Infer spatial dimensions from num_features
        # Assume square spatial dimensions
        H = W = int(num_features ** 0.5)
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
            log_weights = log_weights.view(1, 1, self.out_shape.channels, in_channels, out_reps)

            # Reshape ll for broadcasting: (batch, features, 1, in_c, reps)
            ll = ll.unsqueeze(2)

            # Weighted sum over input channels: logsumexp over dim 3 (in_channels)
            weighted_lls = ll + log_weights
            result = torch.logsumexp(weighted_lls, dim=3)  # (batch, features, out_c, reps)

            return result

        if H % K != 0 or W % K != 0:
            raise ValueError(
                f"Spatial dims ({H}, {W}) must be divisible by kernel_size {K}"
            )

        # Get log weights: (out_c, in_c, k, k, reps)
        log_weights = self.log_weights

        # Reshape ll from (batch, features, in_c, reps) to spatial form
        # (batch, in_c, H, W, reps)
        ll = ll.permute(0, 2, 1, 3)  # (batch, in_c, features, reps)
        ll = ll.view(batch_size, in_channels, H, W, out_reps)

        # Patch the input into KxK blocks
        # (batch, in_c, H//K, K, W//K, K, reps)
        ll = ll.view(batch_size, in_channels, H // K, K, W // K, K, out_reps)
        # Reorder to (batch, in_c, H//K, W//K, K, K, reps)
        ll = ll.permute(0, 1, 2, 4, 3, 5, 6)

        # Make space for out_channels: (batch, 1, in_c, H//K, W//K, K, K, reps)
        ll = ll.unsqueeze(1)

        # Make space in log_weights for spatial dims: (1, out_c, in_c, 1, 1, K, K, reps)
        log_weights = log_weights.unsqueeze(0).unsqueeze(3).unsqueeze(4)

        # Weighted sum over input channels: logsumexp over dim 2 (in_channels)
        weighted_lls = ll + log_weights
        result = torch.logsumexp(weighted_lls, dim=2)  # (batch, out_c, H//K, W//K, K, K, reps)

        # Invert the patch transformation
        # (batch, out_c, H//K, W//K, K, K, reps) -> (batch, out_c, H//K, K, W//K, K, reps)
        result = result.permute(0, 1, 2, 4, 3, 5, 6)
        # Reshape back to (batch, out_c, H, W, reps)
        result = result.contiguous().view(batch_size, self.out_shape.channels, H, W, out_reps)

        # Convert back to (batch, features, out_c, reps)
        result = result.view(batch_size, self.out_shape.channels, num_features, out_reps)
        result = result.permute(0, 2, 1, 3)  # (batch, features, out_c, reps)

        return result


    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
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
        if cache is None:
            cache = Cache()

        # Handle num_samples case
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan")).to(self.device)

        batch_size = data.shape[0]

        # Initialize sampling context
        sampling_ctx = init_default_sampling_context(sampling_ctx, batch_size, data.device)

        num_features = self.in_shape.features

        # Infer spatial dimensions
        H = W = int(num_features ** 0.5)
        K = self.kernel_size

        if H * W != num_features:
            raise ValueError(
                f"SumConv requires square spatial dimensions. Got {num_features} features "
                f"which is not a perfect square."
            )

        if H % K != 0 or W % K != 0:
            raise ValueError(
                f"Spatial dims ({H}, {W}) must be divisible by kernel_size {K}"
            )

        # Expand channel_index and mask to match input features if needed
        current_features = sampling_ctx.channel_index.shape[1]
        if current_features != num_features:
            if current_features == 1:
                expand_sampling_context(sampling_ctx, num_features)
            else:
                # Upsample from parent spatial dims to input spatial dims
                upsample_sampling_context(
                    sampling_ctx,
                    current_height=H // K,
                    current_width=W // K,
                    scale_h=K,
                    scale_w=K,
                )

        channel_idx = sampling_ctx.channel_index  # (batch, H*W)

        # Get logits: (out_c, in_c, k, k, reps)
        logits = self.logits

        # Select repetition
        if sampling_ctx.repetition_idx is not None:
            # logits: (out_c, in_c, k, k, reps) -> select reps
            rep_idx = sampling_ctx.repetition_idx.view(-1, 1, 1, 1, 1)
            rep_idx = rep_idx.expand(batch_size, logits.shape[0], logits.shape[1], K, K)
            logits = logits.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)
            logits = torch.gather(logits, dim=-1, index=rep_idx.unsqueeze(-1)).squeeze(-1)
            # logits: (batch, out_c, in_c, k, k)
        else:
            logits = logits[..., 0]  # (out_c, in_c, k, k)
            logits = logits.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            # logits: (batch, out_c, in_c, k, k)

        # Check for cached likelihoods (conditional sampling)
        input_lls = None
        if (cache is not None
            and "log_likelihood" in cache
            and cache["log_likelihood"].get(self.inputs) is not None):
            input_lls = cache["log_likelihood"][self.inputs]  # (batch, features, in_c, reps)

            # Select repetition
            if sampling_ctx.repetition_idx is not None:
                rep_idx = sampling_ctx.repetition_idx.view(-1, 1, 1, 1)
                rep_idx = rep_idx.expand(-1, input_lls.shape[1], input_lls.shape[2], 1)
                input_lls = torch.gather(input_lls, dim=-1, index=rep_idx).squeeze(-1)
            else:
                input_lls = input_lls[..., 0]
            # input_lls: (batch, H*W, in_c)

            # Reshape to spatial: (batch, H, W, in_c)
            input_lls = input_lls.view(batch_size, H, W, self.in_channels)

        # Reshape channel_idx to spatial: (batch, H, W)
        channel_idx = channel_idx.view(batch_size, H, W)

        # Sample per-position: each pixel position needs its own sample
        # Create position indices for kernel
        # Position within kernel: pixel (i, j) has kernel pos (i % K, j % K)
        row_pos = torch.arange(H, device=data.device).view(1, H, 1).expand(batch_size, H, W)
        col_pos = torch.arange(W, device=data.device).view(1, 1, W).expand(batch_size, H, W)
        k_row = row_pos % K  # (batch, H, W)
        k_col = col_pos % K  # (batch, H, W)

        # logits: (batch, out_c, in_c, k, k)
        # Select logits for each position based on parent channel and kernel position
        # First gather by parent channel: (batch, H, W, in_c, k, k)
        logits_per_pos = logits.permute(0, 3, 4, 1, 2)  # (batch, k, k, out_c, in_c)

        # Index by kernel position
        # k_row, k_col: (batch, H, W)
        # Need to gather from (batch, k, k, out_c, in_c)
        # Flatten kernel dims for gathering
        logits_per_pos = logits_per_pos.view(batch_size, K * K, self.out_shape.channels, self.in_channels)

        # Compute flat kernel index
        k_flat = k_row * K + k_col  # (batch, H, W)

        # Expand for gathering: (batch, H, W, out_c, in_c)
        k_flat_exp = k_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.out_shape.channels, self.in_channels)
        logits_per_pos_exp = logits_per_pos.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W, -1, -1)
        # (batch, K*K, H, W, out_c, in_c) -> swap dims for gather
        logits_per_pos_exp = logits_per_pos_exp.permute(0, 2, 3, 1, 4, 5)  # (batch, H, W, K*K, out_c, in_c)

        # Gather by k_flat
        k_flat_exp2 = k_flat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch, H, W, 1, 1, 1)
        k_flat_exp2 = k_flat_exp2.expand(-1, -1, -1, -1, self.out_shape.channels, self.in_channels)
        selected_logits = torch.gather(logits_per_pos_exp, dim=3, index=k_flat_exp2).squeeze(3)
        # selected_logits: (batch, H, W, out_c, in_c)

        # Now select by parent channel: channel_idx (batch, H, W)
        parent_ch = channel_idx.unsqueeze(-1).unsqueeze(-1)  # (batch, H, W, 1, 1)
        parent_ch = parent_ch.expand(-1, -1, -1, -1, self.in_channels)  # (batch, H, W, 1, in_c)
        selected_logits = torch.gather(selected_logits, dim=3, index=parent_ch).squeeze(3)
        # selected_logits: (batch, H, W, in_c)

        # Compute posterior if we have cached likelihoods
        if input_lls is not None:
            # input_lls: (batch, H, W, in_c)
            log_posterior = selected_logits + input_lls
            log_posterior = F.log_softmax(log_posterior, dim=-1)
        else:
            log_posterior = F.log_softmax(selected_logits, dim=-1)

        # Sample for each position
        log_posterior_flat = log_posterior.view(-1, self.in_channels)
        if is_mpe:
            sampled_channels_flat = torch.argmax(log_posterior_flat, dim=-1)
        else:
            sampled_channels_flat = torch.distributions.Categorical(logits=log_posterior_flat).sample()

        sampled_channels = sampled_channels_flat.view(batch_size, H, W)
        sampled_channels = sampled_channels.view(batch_size, num_features)

        # Update sampling context
        sampling_ctx.channel_index = sampled_channels

        # Sample from input
        self.inputs.sample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

        return data

    def expectation_maximization(
        self,
        data: Tensor,
        bias_correction: bool = True,
        cache: Cache | None = None,
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
            ValueError: If required log-likelihoods are not found in cache.
        """
        if cache is None:
            cache = Cache()

        with torch.no_grad():
            # Get cached log-likelihoods
            input_lls = cache["log_likelihood"].get(self.inputs)
            if input_lls is None:
                raise ValueError("Input log-likelihoods not found in cache. Call log_likelihood first.")

            module_lls = cache["log_likelihood"].get(self)
            if module_lls is None:
                raise ValueError("Module log-likelihoods not found in cache. Call log_likelihood first.")

            # input_lls shape: (batch, features, in_channels, reps)
            # module_lls shape: (batch, features, out_channels, reps)
            # log_weights shape: (out_channels, in_channels, k, k, reps)

            batch_size = input_lls.shape[0]
            num_features = input_lls.shape[1]
            in_channels = input_lls.shape[2]
            out_channels = module_lls.shape[2]
            num_reps = self.out_shape.repetitions

            # Get log gradients from module output
            # grad is set during backward pass or EM routine
            if module_lls.grad is None:
                # If no gradient, use uniform (this happens at the root)
                log_grads = torch.zeros_like(module_lls)
            else:
                log_grads = torch.log(module_lls.grad + 1e-10)

            # Current log weights: (out_c, in_c, k, k, reps)
            # Average over kernel spatial dims for simplicity
            log_weights = self.log_weights.mean(dim=(2, 3))  # (out_c, in_c, reps)

            # Reshape for broadcasting:
            # log_weights: (1, 1, out_c, in_c, reps)
            # log_grads: (batch, features, out_c, 1, reps)
            # input_lls: (batch, features, 1, in_c, reps)
            # module_lls: (batch, features, out_c, 1, reps)

            log_weights = log_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, out_c, in_c, reps)
            log_grads = log_grads.unsqueeze(3)  # (batch, features, out_c, 1, reps)
            input_lls = input_lls.unsqueeze(2)  # (batch, features, 1, in_c, reps)
            module_lls = module_lls.unsqueeze(3)  # (batch, features, out_c, 1, reps)

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
            new_log_weights = log_expectations.unsqueeze(2).unsqueeze(3)  # (out_c, in_c, 1, 1, reps)
            new_log_weights = new_log_weights.expand(-1, -1, k, k, -1)  # (out_c, in_c, k, k, reps)

            # Set new weights
            self.logits.data = new_log_weights.contiguous()

        # Recursively call EM on inputs
        self.inputs.expectation_maximization(data, cache=cache, bias_correction=bias_correction)

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

