from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor, nn

from spflow.exceptions import ScopeError
from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.modules.ops.split_halves import Split
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class BaseProduct(Module, ABC):
    """Base class for product operations in probabilistic circuits.

    Computes joint distributions via factorization. Assumes conditional independence
    between disjoint input scopes. Abstract class requiring mapping method implementations.

    Attributes:
        inputs: Input modules to multiply.
        input_is_split: Whether input is a split operation.
        num_splits: Number of splits if input is split.
        scope: Combined scope of all inputs.
    """

    def __init__(
        self,
        inputs: list[Module] | Module,
    ) -> None:
        """Initialize product module.

        Args:
            inputs: Input module(s) with pairwise disjoint scopes.

        Raises:
            ValueError: No inputs provided.
            ScopeError: Input scopes not pairwise disjoint.
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
        """Map output channel indices to input channel indices.

        Args:
            output_ids: Tensor containing output channel indices.

        Returns:
            Tensor: Mapped input channel indices.
        """
        pass

    @abstractmethod
    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        """Map output mask to input mask.

        Args:
            mask: Output mask tensor.

        Returns:
            Tensor: Mapped input mask tensor.
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
        """Generate samples from product module.

        Args:
            num_samples: Number of samples to generate.
            data: Optional data tensor to store samples.
            is_mpe: Whether to perform most probable explanation.
            cache: Optional cache for computation.
            sampling_ctx: Optional sampling context.

        Returns:
            Tensor: Generated samples.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        # initialize contexts
        if cache is None:
            cache = Cache()
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
        """Marginalize specified variables (must be implemented by subclasses).

        Args:
            marg_rvs: List of variable indices to marginalize.
            prune: Whether to prune the resulting structure.
            cache: Optional cache for computation.

        Returns:
            Optional[BaseProduct | Module]: Marginalized module or None.
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
        """Compute log likelihood.

        Args:
            data: Input data tensor.
            cache: Optional cache for computation.

        Returns:
            Tensor: Log likelihood values.
        """
        pass

    def _get_input_log_likelihoods(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> list[Tensor]:
        """Prepare input log-likelihoods.

        Args:
            data: Input data tensor.
            cache: Optional cache for computation.

        Returns:
            list[Tensor]: List of log-likelihood tensors for each input.
        """
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
                lls.append(ll)

        return lls
