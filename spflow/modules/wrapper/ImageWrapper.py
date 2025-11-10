from typing import Optional, Union, Dict, Any, List

import torch
from torch import Tensor

from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.modules.module import Module
from spflow.modules.wrapper.abstract_wrapper import AbstractWrapper
from spflow.utils.cache import Cache, init_cache


class MarginalizationContext:
    def __init__(self, c: list[int] = None, h: list[int] = None, w: list[int] = None):
        self.c = [] if c is None else c
        self.h = [] if h is None else h
        self.w = [] if w is None else w


class ImageWrapper(AbstractWrapper):
    def __init__(self, module: Module, num_channel: int, height: int, width: int):
        """
        Initialize an ImageWrapper.

        Args:
            module: Module to wrap.
            num_channel: Number of channels in the image.
            height: Height of the image.
            width: Width of the image.
        """
        super().__init__(module=module)
        self.num_channel = num_channel
        self.height = height
        self.width = width
        assert (
            len(module.scope.query) == height * width * num_channel
        ), f"Module out_features {module.out_features} does not match the expected size {height * width * num_channel}."

    def flatten(self, tensor: torch.Tensor):
        assert tensor.dim() == 4, f"Input tensor must be 4-dimensional but got {tensor.dim()}-dimensional."
        assert (
            tensor.shape[1] == self.num_channel
        ), f"Input tensor channel dimension must; match num_channel {self.num_channel} but got {tensor.shape[1]}."

        return tensor.view(tensor.shape[0], -1)

    def to_image_format(self, tensor: torch.Tensor, batch: bool = True):
        if batch:
            assert (
                tensor.dim() == 2
            ), f"Input tensor must be 2-dimensional but got {tensor.dim()}-dimensional."
            return tensor.view(tensor.shape[0], self.num_channel, self.height, self.width)
        else:
            assert (
                tensor.dim() == 1
            ), f"Input tensor must be 1-dimensional but got {tensor.dim()}-dimensional."
            return tensor.view(self.num_channel, self.height, self.width)

    def extra_repr(self):
        return f"C={self.num_channel}, H={self.height}, W={self.width}"

    def expectation_maximization(
        self,
        data: Tensor,
        check_support: bool = True,
        cache: Cache | None = None,
    ) -> None:
        """Performs a single expectation maximization (EM) step for the wrapped module.

        Args:
            data:
                Four-dimensional PyTorch tensor containing the input data.
                Shape: (batch_size, num_channel, height, width).
            check_support:
                Boolean value indicating whether or not if the data is in the support of the leaf distributions.
                Defaults to True.
            cache:
                Optional cache dictionary for memoization.
        """
        cache = init_cache(cache)

        assert data.shape == (
            data.shape[0],
            self.num_channel,
            self.height,
            self.width,
        ), f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."

        data = self.flatten(data)
        self.module.expectation_maximization(data, check_support=check_support, cache=cache)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        check_support: bool = True,
        cache: Cache | None = None,
    ) -> None:
        """Update parameters via maximum likelihood estimation for the wrapped module.

        Args:
            data:
                Four-dimensional PyTorch tensor containing the input data.
                Shape: (batch_size, num_channel, height, width).
            weights:
                Optional sample weights tensor.
            check_support:
                Boolean value indicating whether or not if the data is in the support of the leaf distributions.
                Defaults to True.
            cache:
                Optional cache dictionary for memoization.
        """
        cache = init_cache(cache)

        assert data.shape == (
            data.shape[0],
            self.num_channel,
            self.height,
            self.width,
        ), f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."

        data = self.flatten(data)
        self.module.maximum_likelihood_estimation(
            data, weights=weights, check_support=check_support, cache=cache
        )

    def log_likelihood(
        self,
        data: Tensor,
        check_support: bool = True,
        cache: Cache | None = None,
    ) -> Tensor:
        r"""Computes log-likelihoods for the wrapped module given the data.

        Missing values (i.e., NaN) are marginalized over.

        Args:
            data:
                Four-dimensional PyTorch tensor containing the input data.
                Shape: (batch_size, num_channel, height, width).
            check_support:
                Boolean value indicating whether or not if the data is in the support of the distribution.
                Defaults to True.
            cache:
                Optional cache dictionary for memoization.

        Returns:
            Two-dimensional PyTorch tensor containing the log-likelihoods of the input data.
            Each row corresponds to an input sample.

        Raises:
            ValueError: Data outside of support.
        """
        cache = init_cache(cache)

        assert data.shape == (
            data.shape[0],
            self.num_channel,
            self.height,
            self.width,
        ), f"Data shape must be (batch_size, num_channel, height, width) but got {data.shape}."
        data = self.flatten(data)

        log_prob = self.module.log_likelihood(data, check_support, cache)

        return log_prob

    def _prepare_sample_data(self, num_samples: int | None, data: Tensor | None) -> Tensor:
        """Override to create 4D image tensor.

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
        check_support: bool = True,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
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
            check_support:
                Boolean value indicating whether if the data is in the support of the leaf distributions.
                Defaults to True.
            cache:
                Optional cache dictionary for memoization.
            sampling_ctx:
                Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

        Returns:
            Four-dimensional PyTorch tensor in image format containing the sampled values.
            Shape: (batch_size, num_channel, height, width).
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        cache = init_cache(cache)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        # Flatten data to 2D for processing
        flat_data = self.flatten(data)

        self.module.sample(
            data=flat_data,
            is_mpe=is_mpe,
            check_support=check_support,
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
        """Marginalizes out spatial dimensions from the wrapped module.

        Args:
            marg_ctx:
                MarginalizationContext specifying which dimensions to marginalize (channels, height, width).
            prune:
                Boolean value indicating whether to prune the structure after marginalization.
                Defaults to True.
            cache:
                Optional cache dictionary for memoization.

        Returns:
            New ImageWrapper with marginalized module and adjusted dimensions, or None if module is fully marginalized.
        """
        cache = init_cache(cache)

        marg_rvs = []
        scope = torch.tensor(self.scope.query)

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
