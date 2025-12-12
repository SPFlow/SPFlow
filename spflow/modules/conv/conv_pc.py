"""Convolutional Probabilistic Circuit.

Provides ConvPc, a multi-layer architecture that stacks alternating
SumConv and ProdConv layers on top of a leaf distribution.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from spflow.modules.conv.prod_conv import ProdConv
from spflow.modules.conv.sum_conv import SumConv
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.sums import Sum
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext


def compute_non_overlapping_kernel_and_padding(
    H_data: int, W_data: int, H_target: int, W_target: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute kernel size and padding for non-overlapping convolution.

    Computes kernel size and padding such that a single F.conv2d with
    stride=kernel_size and dilation=1 transforms the input to the target size.

    Args:
        H_data: Input height.
        W_data: Input width.
        H_target: Target output height.
        W_target: Target output width.

    Returns:
        Tuple of (kernel_size, padding) where:
            kernel_size: (kH, kW)
            padding: (pH, pW)

    Raises:
        ValueError: If any dimension is non-positive.
    """
    if H_data <= 0 or W_data <= 0 or H_target <= 0 or W_target <= 0:
        raise ValueError("All dimensions must be positive.")

    # Compute required kernel sizes
    kH = math.ceil(H_data / H_target)
    kW = math.ceil(W_data / W_target)

    # Compute padding needed to make input + 2*padding divisible by kernel
    padded_H = kH * H_target
    padded_W = kW * W_target

    total_pad_H = max(padded_H - H_data, 0)
    total_pad_W = max(padded_W - W_data, 0)

    pH = total_pad_H // 2
    pW = total_pad_W // 2

    return (kH, kW), (pH, pW)


class ConvPc(Module):
    """Convolutional Probabilistic Circuit.

    Builds a multi-layer circuit with alternating ProdConv and SumConv layers
    on top of a leaf distribution. The architecture progressively reduces
    spatial dimensions while learning mixture weights at each level.

    The layer ordering is: Leaf -> ProdConv -> SumConv -> ProdConv -> SumConv -> ... -> Root Sum

    Layers are constructed top-down (from root to leaves), then reversed for
    proper bottom-up evaluation order.

    Attributes:
        leaf (Module): Leaf distribution module.
        root (Sum): Final sum layer producing scalar output per sample.
    """

    def __init__(
        self,
        leaf: Module,
        input_height: int,
        input_width: int,
        channels: int,
        depth: int,
        kernel_size: int = 2,
        num_repetitions: int = 1,
        use_sum_conv: bool = False,
    ) -> None:
        """Create a ConvPc for image modeling.

        Args:
            leaf: Leaf distribution module (e.g., Normal over pixels).
            input_height: Height of input image.
            input_width: Width of input image.
            channels: Number of channels per sum layer.
            depth: Number of (ProdConv, SumConv) layer pairs.
            kernel_size: Kernel size for pooling (default 2x2).
            num_repetitions: Number of independent repetitions.
            use_sum_conv: If True, use SumConv layers with kernel-based spatial
                weights. If False (default), use regular Sum layers that treat
                features independently without spatial awareness.

        Raises:
            ValueError: If depth < 1.
        """
        super().__init__()
        self.use_sum_conv = use_sum_conv

        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        self.input_height = input_height
        self.input_width = input_width
        self.kernel_size = kernel_size
        self.depth = depth

        # Build layers top-down: start from root (1x1) and work down
        # Top-down order: Sum (root) -> ProdConv -> SumConv -> ProdConv -> SumConv -> ... -> Leaf
        # We'll build a list of layer specs top-down, then reverse and construct bottom-up
        layer_specs = []

        # Add root sum layer: 1x1 spatial, reduces channels to 1
        layer_specs.append(("sum_root", {"out_channels": 1}))

        # Build from top (1x1) down to target spatial size
        # Each depth level adds: ProdConv (spatial expansion) -> SumConv (channel mixing)
        h, w = 1, 1
        for i in reversed(range(depth)):
            # ProdConv expands spatial dims by kernel_size
            layer_specs.append(
                (
                    "prod",
                    {"kernel_size": kernel_size},
                )
            )
            # SumConv mixes channels at this spatial level
            h, w = h * kernel_size, w * kernel_size
            layer_specs.append(
                (
                    "sum",
                    {
                        "out_channels": channels,
                        "kernel_size": kernel_size,
                    },
                )
            )

        # Now h, w is the target spatial size at the bottom of the conv layers
        # We need a final ProdConv to reduce from input_height/width to h/w

        # Compute kernel and padding for the bottom ProdConv layer
        (kh, kw), (ph, pw) = compute_non_overlapping_kernel_and_padding(
            H_data=input_height,
            W_data=input_width,
            H_target=h,
            W_target=w,
        )
        layer_specs.append(
            (
                "prod_bottom",
                {"kernel_size_h": kh, "kernel_size_w": kw, "padding_h": ph, "padding_w": pw},
            )
        )

        # Reverse the specs so we build bottom-up (leaf -> ... -> root)
        layer_specs = list(reversed(layer_specs))

        # Now construct layers bottom-up, connecting via .inputs
        # Bottom-up order: Leaf -> ProdConv -> SumConv -> ProdConv -> SumConv -> ... -> Sum (root)
        current_input = leaf

        for layer_type, params in layer_specs:
            if layer_type == "prod_bottom":
                current_input = ProdConv(
                    inputs=current_input,
                    kernel_size_h=params["kernel_size_h"],
                    kernel_size_w=params["kernel_size_w"],
                    padding_h=params.get("padding_h", 0),
                    padding_w=params.get("padding_w", 0),
                )
            elif layer_type == "sum":
                if self.use_sum_conv:
                    current_input = SumConv(
                        inputs=current_input,
                        out_channels=params["out_channels"],
                        kernel_size=params["kernel_size"],
                        num_repetitions=num_repetitions,
                    )
                else:
                    current_input = Sum(
                        inputs=current_input,
                        out_channels=params["out_channels"],
                        num_repetitions=num_repetitions,
                    )
            elif layer_type == "prod":
                current_input = ProdConv(
                    inputs=current_input,
                    kernel_size_h=params["kernel_size"],
                    kernel_size_w=params["kernel_size"],
                )
            elif layer_type == "sum_root":
                # Final root sum to produce single output (1x1 spatial, 1 channel)
                current_input = Sum(
                    inputs=current_input,
                    out_channels=params["out_channels"],
                    num_repetitions=num_repetitions,
                )

        self.inputs = current_input

        # Add repetition mixing layer if num_repetitions > 1
        if num_repetitions > 1:
            self.inputs = RepetitionMixingLayer(
                inputs=self.inputs,
                out_channels=1,
                num_repetitions=num_repetitions,
            )

        # Scope and shape
        self.scope = leaf.scope

        leaf_shape = leaf.out_shape
        self.in_shape = leaf_shape
        self.out_shape = ModuleShape(
            features=1,
            channels=1,
            repetitions=1,  # Always 1 after mixing layer
        )

    @property
    def feature_to_scope(self) -> np.ndarray:
        """Single output feature with full scope."""
        return self.inputs.feature_to_scope

    def extra_repr(self) -> str:
        return (
            f"input=({self.input_height}, {self.input_width}), "
            f"depth={self.depth}, kernel_size={self.kernel_size}"
        )

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood through all layers.

        Args:
            data: Input data of shape (batch_size, num_pixels).
            cache: Cache for intermediate computations.

        Returns:
            Tensor: Log-likelihood of shape (batch, 1, 1, reps).
        """
        if cache is None:
            cache = Cache()

        # Forward through root, which recursively calls inputs
        # Chain: root -> SumConv -> ProdConv -> ... -> leaf
        return self.inputs.log_likelihood(data, cache=cache)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples by sampling top-down through layers.

        Delegates sampling to the root module (RepetitionMixingLayer when
        num_repetitions > 1, or Sum when num_repetitions == 1), which then
        recursively propagates sampling to the leaf.

        Args:
            num_samples: Number of samples to generate.
            data: Data tensor with NaN values to fill with samples.
            is_mpe: Whether to perform maximum a posteriori estimation.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Tensor: Sampled values of shape (num_samples, num_pixels).
        """
        if cache is None:
            cache = Cache()

        # Handle num_samples case
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full(
                (num_samples, len(self.scope.query)), float("nan")
            ).to(self.device)

        # Delegate to root (RepetitionMixingLayer or Sum)
        # which handles channel/repetition sampling internally
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
        """Perform EM update throughout the circuit.

        Args:
            data: Input data tensor.
            bias_correction: Whether to apply bias correction.
            cache: Optional cache with log-likelihoods.
        """
        if cache is None:
            cache = Cache()

        # EM on root (which chains to all layers)
        self.inputs.expectation_maximization(data, cache=cache, bias_correction=bias_correction)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> ConvPc | Module | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variable indices to marginalize.
            prune: Whether to prune unnecessary nodes.
            cache: Optional cache for storing intermediate results.

        Returns:
            ConvPc | Module | None: Marginalized module or None if fully marginalized.
        """
        # For ConvPc, marginalization is complex due to the layered architecture
        # Delegate to root which handles it recursively
        return self.inputs.marginalize(marg_rvs, prune=prune, cache=cache)

