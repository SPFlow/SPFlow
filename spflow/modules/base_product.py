from __future__ import annotations

from abc import ABC, abstractmethod
from spflow.modules.ops.split_halves import Split

import torch
from torch import Tensor, nn
from typing import Optional, Dict, Any

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.modules.module import Module
from spflow.utils.cache import Cache, init_cache


class BaseProduct(Module, ABC):
    r"""
    Base class for the modules OuterProduct and ElementwiseProduct.
    """

    def __init__(
        self,
        inputs: list[Module] | Module,
    ) -> None:
        r"""Initializes ``BaseProduct`` object.

        Args:
            inputs:
                Single input module or list of modules.

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__()

        # Obtain number of splits and check input type
        if isinstance(inputs, Split):
            inputs = [inputs]
            self.input_is_split = True
            self.num_splits = inputs[0].num_splits
        else:
            self.input_is_split = False
            if inputs[0].out_features == 1:
                self.num_splits = 1
            else:
                self.num_splits = None

        if not inputs:
            raise ValueError(f"'{self.__class__.__name__}' requires at least one input to be specified.")

        self.inputs = nn.ModuleList(inputs)

        # Check that scopes are disjoint
        if not Scope.all_pairwise_disjoint([inp.scope for inp in self.inputs]):
            raise ScopeError("Input scopes must be disjoint.")

        self._max_out_channels = max(inp.out_channels for inp in self.inputs)

        # Join all scopes
        scope = self.inputs[0].scope
        for inp in self.inputs[1:]:
            scope = scope.join(inp.scope)

        self.scope = scope

        self.num_repetitions = self.inputs[0].num_repetitions

    @abstractmethod
    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        r"""Map output ids to input ids.

        Args:
            output_ids: Output ids.

        Returns:
            Mapped input ids.
        """
        pass

    @abstractmethod
    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        r"""Map output mask to input mask.

        Args:
            mask: Output mask.

        Returns:
            Mapped input mask.
        """
        pass

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}"

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        """Generate samples from the product module.

        Args:
            num_samples: Number of samples to generate.
            data: The data tensor to populate with samples.
            is_mpe: Whether to use maximum probability estimation instead of sampling.
            cache: Optional cache dictionary for intermediate results.
            sampling_ctx: Optional sampling context.

        Returns:
            The data tensor populated with samples.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        # initialize contexts
        cache = init_cache(cache)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

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
            inp.sample(
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
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variables to marginalize over.
            prune: Whether to prune the structure.
            cache: Optional cache dictionary.

        Returns:
            The marginalized module or None if fully marginalized.
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
        """Compute log P(data | module).

        Args:
            data: The data tensor.
            cache: Optional cache dictionary.

        Returns:
            Log likelihood tensor.
        """
        pass

    def _get_input_log_likelihoods(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> list[Tensor]:
        """
        Prepare the input log-likelihoods for the product module.

        Args:
            data: The data tensor.
            cache: The cache dictionary.
        """

        log_cache = None
        if cache is not None:
            log_cache = cache.setdefault("log_likelihood", {})

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
                if log_cache is not None:
                    log_cache[inp] = ll
                lls.append(ll)

        return lls
