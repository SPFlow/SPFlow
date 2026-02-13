"""Convolutional product layer for probabilistic circuits.

Provides ProdConv, which computes products over spatial patches (sums in log-space),
reducing spatial dimensions while aggregating scopes.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from spflow.exceptions import ShapeError
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext
from spflow.modules.conv.utils import upsample_sampling_context


class ProdConv(Module):
    """Convolutional product layer for probabilistic circuits.

    Computes products over spatial patches, reducing spatial dimensions by the
    kernel size factor. This is equivalent to summing log-likelihoods within patches.
    No learnable parameters.

    Scopes are aggregated per patch: a 2×2 patch containing Scope(0), Scope(1),
    Scope(2), Scope(3) produces Scope([0,1,2,3]).

    Attributes:
        inputs (Module): Input module providing log-likelihoods.
        kernel_size_h (int): Kernel height.
        kernel_size_w (int): Kernel width.
        padding_h (int): Padding in height dimension.
        padding_w (int): Padding in width dimension.
    """

    def __init__(
        self,
        inputs: Module,
        kernel_size_h: int,
        kernel_size_w: int,
        padding_h: int = 0,
        padding_w: int = 0,
    ) -> None:
        """Create a ProdConv module for spatial product operations.

        Args:
            inputs: Input module providing log-likelihoods with spatial structure.
            kernel_size_h: Height of the pooling kernel.
            kernel_size_w: Width of the pooling kernel.
            padding_h: Padding in height dimension (added on both sides).
            padding_w: Padding in width dimension (added on both sides).

        Raises:
            ValueError: If kernel sizes are < 1.
        """
        super().__init__()

        if kernel_size_h < 1:
            raise ValueError(f"kernel_size_h must be >= 1, got {kernel_size_h}")
        if kernel_size_w < 1:
            raise ValueError(f"kernel_size_w must be >= 1, got {kernel_size_w}")

        self.inputs = inputs
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.padding_h = padding_h
        self.padding_w = padding_w

        # Infer input shape
        input_shape = self.inputs.out_shape
        in_features = input_shape.features

        # Infer spatial dimensions (assumes square for now, can be extended)
        in_h = in_w = int(np.sqrt(in_features))
        assert in_h * in_w == in_features, f"Features {in_features} must be a perfect square"

        # Compute output spatial dimensions
        padded_h = in_h + 2 * padding_h
        padded_w = in_w + 2 * padding_w
        out_h = padded_h // kernel_size_h
        out_w = padded_w // kernel_size_w

        self._input_h = in_h
        self._input_w = in_w
        self._output_h = out_h
        self._output_w = out_w

        # Shape computation
        self.in_shape = input_shape
        self.out_shape = ModuleShape(
            features=out_h * out_w,
            channels=input_shape.channels,
            repetitions=input_shape.repetitions,
        )

        # Compute aggregated scope
        # Each output position covers a kernel_size_h x kernel_size_w patch
        self._compute_scope()

    def _compute_scope(self) -> None:
        """Compute the aggregated scope for this layer."""
        # Get input scopes
        input_scopes = self.inputs.feature_to_scope

        # For now, compute overall scope as union of all input scopes
        all_rvs = set()
        for scope in input_scopes.flatten():
            all_rvs.update(scope.query)
        self.scope = Scope(sorted(all_rvs))

    @property
    def feature_to_scope(self) -> np.ndarray:
        """Aggregated scopes per output feature.

        Each output feature's scope is the join of all input scopes within its patch.

        Returns:
            np.ndarray: 2D array of Scope objects (features, repetitions).
        """
        input_f2s = self.inputs.feature_to_scope  # (in_features, reps)
        in_h, in_w = self._input_h, self._input_w
        out_h, out_w = self._output_h, self._output_w
        kh, kw = self.kernel_size_h, self.kernel_size_w
        ph, pw = self.padding_h, self.padding_w
        num_reps = input_f2s.shape[1]

        result = np.empty((out_h * out_w, num_reps), dtype=object)

        for r in range(num_reps):
            # Reshape input scopes to spatial: (in_h, in_w)
            input_scopes_2d = input_f2s[:, r].reshape(in_h, in_w)

            out_idx = 0
            for oh in range(out_h):
                for ow in range(out_w):
                    # Compute input patch bounds
                    start_h = oh * kh - ph
                    end_h = start_h + kh
                    start_w = ow * kw - pw
                    end_w = start_w + kw

                    # Collect scopes from valid positions (not padding)
                    patch_scopes = []
                    for ih in range(max(0, start_h), min(in_h, end_h)):
                        for iw in range(max(0, start_w), min(in_w, end_w)):
                            patch_scopes.append(input_scopes_2d[ih, iw])

                    # Join all scopes in the patch
                    if patch_scopes:
                        result[out_idx, r] = Scope.join_all(patch_scopes)
                    else:
                        # Edge case: entire patch is padding
                        result[out_idx, r] = Scope([])

                    out_idx += 1

        return result

    def extra_repr(self) -> str:
        return (
            f"kernel=({self.kernel_size_h}, {self.kernel_size_w}), "
            f"padding=({self.padding_h}, {self.padding_w})"
        )

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood by summing within patches.

        Uses depthwise convolution with ones kernel to efficiently sum
        log-probabilities within patches.

        Args:
            data: Input data of shape (batch_size, num_features).
            cache: Cache for intermediate computations.

        Returns:
            Tensor: Log-likelihood of shape (batch, out_features, channels, reps).
        """
        # Get input log-likelihoods: (batch, features, channels, reps)
        ll = self.inputs.log_likelihood(data, cache=cache)

        batch_size = ll.shape[0]
        in_features = ll.shape[1]
        channels = ll.shape[2]
        reps = ll.shape[3]

        in_h, in_w = self._input_h, self._input_w
        out_h, out_w = self._output_h, self._output_w
        kh, kw = self.kernel_size_h, self.kernel_size_w
        ph, pw = self.padding_h, self.padding_w

        # Merge batch and reps for efficient depthwise convolution.
        ll = rearrange(ll, "b (h w) ci r -> (b r) ci h w", h=in_h, w=in_w)

        # Apply depthwise convolution with ones kernel
        # This sums values within each patch
        ones_kernel = torch.ones(channels, 1, kh, kw, device=ll.device, dtype=ll.dtype)
        result = F.conv2d(ll, ones_kernel, stride=(kh, kw), padding=(ph, pw), groups=channels)

        # Convert back to SPFlow format: (batch, out_features, channels, reps)
        result = rearrange(result, "(b r) ci oh ow -> b (oh ow) ci r", b=batch_size, r=reps)

        return result

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        """Generate samples by delegating to input.

        ProdConv has no learnable parameters, so sampling simply expands
        the sampling context to match input features and delegates.

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

        # Expand channel_index and mask to match input features
        in_features = self.in_shape.features
        out_features = self.out_shape.features

        current_features = sampling_ctx.channel_index.shape[1]

        if current_features == in_features:
            pass
        elif current_features == out_features:
            # Deterministic spatial upsampling from ProdConv output grid to input grid.
            upsample_sampling_context(
                sampling_ctx,
                current_height=self._output_h,
                current_width=self._output_w,
                scale_h=self.kernel_size_h,
                scale_w=self.kernel_size_w,
            )
            upsampled_features = int(sampling_ctx.channel_index.shape[1])
            expected_upsampled_features = (
                self._output_h * self.kernel_size_h * self._output_w * self.kernel_size_w
            )
            if upsampled_features != expected_upsampled_features:
                raise ShapeError(
                    "ProdConv.sample produced unexpected feature width after upsampling: "
                    f"got {upsampled_features}, expected {expected_upsampled_features}."
                )

            # Handle explicit padding case: trim padded spatial positions to input size.
            if upsampled_features > in_features:
                expected_padded_features = (
                    (self._input_h + 2 * self.padding_h) * (self._input_w + 2 * self.padding_w)
                )
                if self.padding_h == 0 and self.padding_w == 0:
                    raise ShapeError(
                        "ProdConv.sample received oversized upsampled feature width without padding: "
                        f"got {upsampled_features}, input features are {in_features}."
                    )
                if upsampled_features != expected_padded_features:
                    raise ShapeError(
                        "ProdConv.sample only trims in the exact padding case. "
                        f"Got upsampled width {upsampled_features}, expected padded width "
                        f"{expected_padded_features}, input width {in_features}."
                    )
                channel_idx = sampling_ctx.channel_index[:, :in_features].contiguous()
                mask = sampling_ctx.mask[:, :in_features].contiguous()
                sampling_ctx.update(channel_index=channel_idx, mask=mask)
            elif upsampled_features < in_features:
                raise ShapeError(
                    "ProdConv.sample upsampling produced too few features for input routing: "
                    f"got {upsampled_features}, expected at least {in_features}."
                )
        else:
            raise ShapeError(
                "ProdConv.sample received incompatible sampling context feature width: "
                f"got {current_features}, expected {out_features} or {in_features}."
            )
        sampling_ctx.require_feature_width(expected_features=in_features)

        # Sample from input
        self.inputs._sample(
            data=data,
            is_mpe=is_mpe,
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
        """EM step (delegates to input, no learnable parameters).

        Args:
            data: Input data tensor for EM step.
            bias_correction: Whether to apply bias correction.
            cache: Optional cache for storing intermediate results.
        """
        # Product has no learnable parameters, delegate to input
        self.inputs._expectation_maximization_step(data, cache=cache, bias_correction=bias_correction)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> ProdConv | Module | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variable indices to marginalize.
            prune: Whether to prune unnecessary nodes.
            cache: Optional cache for storing intermediate results.

        Returns:
            ProdConv | Module | None: Marginalized module or None if fully marginalized.
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

        # For now, return a new ProdConv with marginalized input
        # Note: This is a simplified implementation
        return ProdConv(
            inputs=marg_input,
            kernel_size_h=self.kernel_size_h,
            kernel_size_w=self.kernel_size_w,
            padding_h=self.padding_h,
            padding_w=self.padding_w,
        )
