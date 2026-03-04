from typing import Optional, Union

import torch
from einops import rearrange
from torch import Tensor

from spflow.exceptions import ShapeError, StructureError
from spflow.modules.module import Module
from spflow.modules.wrapper.base import Wrapper
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import LeafParamRecord, SamplingContext


class MarginalizationContext:
    """Context for spatial marginalization in image data.

    Provides a structured way to specify which spatial dimensions
    (channels, height, width) to marginalize when working
    with image data in probabilistic circuits.

    Attributes:
        c (list[int]): Channel indices to marginalize.
        h (list[int]): Height indices to marginalize.
        w (list[int]): Width indices to marginalize.
    """

    def __init__(self, c: list[int] = None, h: list[int] = None, w: list[int] = None):
        """Initialize marginalization context.

        Args:
            c: Channel indices to marginalize.
            h: Height indices to marginalize.
            w: Width indices to marginalize.
        """
        self.c = [] if c is None else c
        self.h = [] if h is None else h
        self.w = [] if w is None else w


class ImageWrapper(Wrapper):
    """Wrapper for adapting SPFlow modules to image data format.

    Provides automatic conversion between 2D flattened tensors used by
    SPFlow modules and 4D image tensors (batch, channels, height, width)
    commonly used in computer vision applications. The wrapper automatically validates
    image dimensions against module scope, handles conversion between 2D and 4D tensor
    formats, and supports all standard SPFlow operations with image data while maintaining
    the spatial structure of image data and enabling the use of standard SPFlow modules.

    Attributes:
        module (Module): Wrapped SPFlow module.
        num_channel (int): Number of image channels.
        height (int): Image height in pixels.
        width (int): Image width in pixels.
    """

    def __init__(self, module: Module, num_channel: int, height: int, width: int):
        """Initialize image wrapper.

        Creates a wrapper that adapts SPFlow modules for image data
        with automatic format conversion and validation.

        Args:
            module: SPFlow module to wrap.
            num_channel: Number of channels in image.
            height: Height of image in pixels.
            width: Width of image in pixels.

        Raises:
            StructureError: If module scope size doesn't match image dimensions.
        """
        super().__init__(module=module)
        self.num_channel = num_channel
        self.height = height
        self.width = width
        if len(module.scope.query) != height * width * num_channel:
            raise StructureError(
                f"Module out_features {module.out_shape.features} does not match the expected size "
                f"{height * width * num_channel}."
            )

    def flatten(self, tensor: torch.Tensor):
        """Convert 4D image tensor to 2D flattened tensor.

        Args:
            tensor: 4D tensor of shape (batch, channels, height, width).

        Returns:
            2D flattened tensor of shape (batch, channels*height*width).

        Raises:
            ShapeError: If tensor is not 4D or channel dimension mismatch.
        """
        if tensor.dim() != 4:
            raise ShapeError(f"Input tensor must be 4-dimensional but got {tensor.dim()}-dimensional.")
        if tensor.shape[1] != self.num_channel:
            raise ShapeError(
                f"Input tensor channel dimension must match num_channel {self.num_channel} but got "
                f"{tensor.shape[1]}."
            )

        return rearrange(tensor, "b c h w -> b (c h w)")

    def to_image_format(self, tensor: torch.Tensor, batch: bool = True):
        """Convert 2D tensor to 4D image format.

        Args:
            tensor: 2D tensor to reshape.
            batch: Whether to include batch dimension.

        Returns:
            4D tensor in image format.

        Raises:
            ShapeError: If tensor dimensions are incompatible.
        """
        if batch:
            if tensor.dim() != 2:
                raise ShapeError(f"Input tensor must be 2-dimensional but got {tensor.dim()}-dimensional.")
            num_channels = self.num_channel
            image_height = self.height
            image_width = self.width
            return rearrange(
                tensor,
                "b (c h w) -> b c h w",
                c=num_channels,
                h=image_height,
                w=image_width,
            )
        else:
            if tensor.dim() != 1:
                raise ShapeError(f"Input tensor must be 1-dimensional but got {tensor.dim()}-dimensional.")
            num_channels = self.num_channel
            image_height = self.height
            image_width = self.width
            return rearrange(
                tensor,
                "(c h w) -> c h w",
                c=num_channels,
                h=image_height,
                w=image_width,
            )

    def extra_repr(self):
        return f"C={self.num_channel}, H={self.height}, W={self.width}"

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        """Performs a single expectation maximization (EM) step for the wrapped module.

        Args:
            data:
                Four-dimensional PyTorch tensor containing the input data.
                Shape: (batch_size, num_channel, height, width).
            bias_correction:
                Whether to apply bias correction in leaf updates.
            cache:
                Optional cache dictionary for memoization.
        """

        if data.shape != (data.shape[0], self.num_channel, self.height, self.width):
            raise ShapeError(
                f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."
            )

        data = self.flatten(data)
        self.module._expectation_maximization_step(data, bias_correction=bias_correction, cache=cache)

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        r"""Computes log-likelihoods for the wrapped module given the data.

        Missing values (i.e., NaN) are marginalized over.

        Args:
            data:
                Four-dimensional PyTorch tensor containing the input data.
                Shape: (batch_size, num_channel, height, width).
            cache:
                Optional cache dictionary for memoization.

        Returns:
            Two-dimensional PyTorch tensor containing the log-likelihoods of the input data.
            Each row corresponds to an input sample.

        Raises:
            ValueError: Data outside of support.
        """

        if data.shape != (data.shape[0], self.num_channel, self.height, self.width):
            raise ShapeError(
                f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."
            )
        data = self.flatten(data)

        log_prob = self.module.log_likelihood(data, cache=cache)

        return log_prob

    def _prepare_sample_data(self, num_samples: int | None, data: Tensor | None) -> Tensor:
        """Create 4D image tensor for sampling.

        Args:
            num_samples: Number of samples to generate.
            data: Existing data tensor.

        Returns:
            Data tensor ready for sampling (4D image format).

        Raises:
            ValueError: If both num_samples and data are provided but num_samples != data.shape[0].
        """
        # Same validation as base
        if data is not None and num_samples is not None:
            if data.shape[0] != num_samples:
                raise ValueError(
                    f"num_samples ({num_samples}) must match data.shape[0] ({data.shape[0]}) or be None"
                )

        # Create 4D image tensor if needed
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full(
                (num_samples, self.num_channel, self.height, self.width),
                float("nan"),
            ).to(self.device)

        return data

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        return_leaf_params: bool = False,
    ) -> Tensor | tuple[Tensor, list[LeafParamRecord]]:
        r"""Samples from the wrapped module, returning results in image format.

        Args:
            num_samples:
                Number of samples to generate.
            data:
                Four-dimensional PyTorch tensor containing the input data.
                Shape: (batch_size, num_channel, height, width).
            is_mpe:
                Boolean value indicating whether to perform maximum a posteriori estimation (MPE).
                Defaults to False.
            cache:
                Optional cache dictionary for memoization.
            return_leaf_params:
                Whether to return leaf distribution parameters collected during traversal.

        Returns:
            Four-dimensional PyTorch tensor in image format containing the sampled values.
            Shape: (batch_size, num_channel, height, width).
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)
        if cache is None:
            cache = Cache()

        sampling_ctx = SamplingContext(
            num_samples=data.shape[0],
            device=data.device,
            is_mpe=is_mpe,
            return_leaf_params=return_leaf_params,
        )

        # Flatten data to 2D for processing
        flat_data = self.flatten(data)

        self.module._sample(
            data=flat_data,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )
        image = self.to_image_format(flat_data)
        if return_leaf_params:
            return image, sampling_ctx.leaf_param_records()
        return image

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        flat_data = self.flatten(data)
        self.module._sample(
            data=flat_data,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )
        return self.to_image_format(flat_data)


    def marginalize(
        self,
        marg_ctx: MarginalizationContext,
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Union[None, "ImageWrapper"]:
        """Marginalize out spatial dimensions from the wrapped module.

        Args:
            marg_ctx: MarginalizationContext specifying which dimensions to marginalize.
            prune: Whether to prune the structure after marginalization.
            cache: Optional cache dictionary for memoization.

        Returns:
            New ImageWrapper with marginalized module and adjusted dimensions,
                or None if module is fully marginalized.
        """

        marg_rvs = []
        scope = torch.as_tensor(self.scope.query, dtype=torch.long)

        scope = self.to_image_format(scope, batch=False)

        reduction_h = len(marg_ctx.h)
        reduction_w = len(marg_ctx.w)
        reduction_c = len(marg_ctx.c)
        for height_id in marg_ctx.h:
            marg_rvs.extend(scope[:, height_id, :].flatten().tolist())
        for width_id in marg_ctx.w:
            marg_rvs.extend(scope[:, :, width_id].flatten().tolist())
        for channel_id in marg_ctx.c:
            marg_rvs.extend(scope[channel_id, :, :].flatten().tolist())

        marg_input = self.module.marginalize(marg_rvs, prune=prune, cache=cache)

        if marg_input is None:
            return None
        else:
            return ImageWrapper(
                module=marg_input,
                height=self.height - reduction_h,
                width=self.width - reduction_w,
                num_channel=self.num_channel - reduction_c,
            )
