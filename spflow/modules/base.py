"""Contains the abstract ``Module`` class for SPFlow modules in the ``base`` backend.

All valid SPFlow modules in the ``base`` backend should inherit from this class or a subclass of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.utils.cache import Cache, init_cache


class Module(nn.Module, ABC):
    """Abstract module class for building graph-based models."""

    def __init__(self) -> None:
        """Initializes the module."""
        super().__init__()
        self.inputs: nn.ModuleList = nn.ModuleList()

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Returns the number of output features of the module."""
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Returns the number of output channels of the module."""
        pass

    @property
    def scope(self) -> Scope:
        """Returns the scope of the module."""
        return self._scope

    @scope.setter
    def scope(self, scope: Scope):
        """Sets the scope of the module."""
        self._scope = scope

    @property
    @abstractmethod
    def feature_to_scope(self) -> list[Scope]:
        """Returns the mapping from features to scopes."""
        pass

    @property
    def device(self):
        """
        Returns the device of the model. If the model parameters are on different devices,
        it returns the device of the first parameter. If the model has no parameters,
        it returns 'cpu' as the default device.
        """
        try:
            return next(iter(self.parameters())).device
        except StopIteration:
            return torch.device("cpu")

    def _prepare_sample_data(self, num_samples: int | None, data: Tensor | None) -> Tensor:
        """Prepare data tensor for sampling with validation.

        Validates num_samples and data parameters, creates data tensor if needed.

        Args:
            num_samples: Number of samples to generate.
            data: Existing data tensor.

        Returns:
            Data tensor ready for sampling.

        Raises:
            ValueError: If both num_samples and data are provided but num_samples != data.shape[0].
        """
        # Strict validation
        if data is not None and num_samples is not None:
            if data.shape[0] != num_samples:
                raise ValueError(
                    f"num_samples ({num_samples}) must match data.shape[0] ({data.shape[0]}) " f"or be None"
                )

        # Create data tensor if needed
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan")).to(self.device)

        return data

    @abstractmethod
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log P(data | module).

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary for intermediate results.

        Returns:
            Log-likelihood values.
        """
        pass

    @abstractmethod
    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples from the module.

        Args:
            num_samples: Number of samples to generate.
            data: Optional data tensor for structured sampling.
            is_mpe: Whether to perform maximum a posteriori estimation.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Sampled values.
        """
        pass

    def sample_with_evidence(
        self,
        evidence: Tensor,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        r"""Samples from modules backend with evidence.

        This is effectively calling log_likelihood to populate the dispatch context cache and then sampling from the module.

        Args:
            evidence:
                Evidence tensor.
            is_mpe:
                Boolean value indicating whether to perform maximum a posteriori estimation (MPE).
                Defaults to False.
            cache:
                Optional cache dictionary to reuse across calls.
            sampling_ctx:
                Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

        Returns:
            Two-dimensional NumPy array containing the sampled values.
            Each row corresponds to a sample.
        """
        cache = init_cache(cache)

        self.log_likelihood(evidence, cache=cache)

        return self.sample(
            data=evidence,
            is_mpe=is_mpe,
            sampling_ctx=sampling_ctx,
            cache=cache,
        )

    def expectation_maximization(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> None:
        """Expectation-maximization step.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary.
        """
        cache = init_cache(cache)

        for input_module in self.inputs:
            input_module.expectation_maximization(data, cache=cache)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        cache: Cache | None = None,
    ) -> None:
        """Update parameters via maximum likelihood estimation.

        Args:
            data: Input data tensor.
            weights: Optional sample weights.
            cache: Optional cache dictionary.
        """
        cache = init_cache(cache)

        for input_module in self.inputs:
            input_module.maximum_likelihood_estimation(
                data,
                weights=weights,
                cache=cache,
            )

    @abstractmethod
    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variables to marginalize.
            prune: Whether to prune the module.
            cache: Optional cache dictionary.

        Returns:
            Marginalized module or None.
        """
        pass

    def forward(self, data: Tensor, cache: Cache | None = None):
        """Forward pass is simply the log-likelihood function."""
        return self.log_likelihood(data, cache=cache)

    def extra_repr(self) -> str:
        return f"D={self.out_features}, C={self.out_channels}, R={self.num_repetitions}"

    def to_str(
        self,
        format: str = "tree",
        max_depth: int | None = None,
        show_params: bool = True,
        show_scope: bool = True,
    ) -> str:
        """
        Convert this module to a readable string representation.

        This method provides visualization formats for understanding module structure.

        Args:
            format: Visualization format, one of:
                - "tree": ASCII tree view (default, recommended)
                - "pytorch": Default PyTorch format
            max_depth: Maximum depth to display (None = unlimited). Only applies to tree format.
            show_params: Whether to show parameter shapes (Sum weights, etc.). Only applies to tree format.
            show_scope: Whether to show scope information. Only applies to tree format.

        Returns:
            String representation of the module.

        Examples:
            >>> leaves = Normal(scope=Scope([0, 1]), out_channels=2)
            >>> model = Sum(inputs=leaves, out_channels=3)
            >>> print(model.to_str())  # Tree view (default)
            >>> print(model.to_str(format="pytorch"))  # PyTorch format
            >>> print(model.to_str(max_depth=2))  # Limit depth
        """
        from spflow.utils.module_display import module_to_str

        return module_to_str(
            self,
            format=format,
            max_depth=max_depth,
            show_params=show_params,
            show_scope=show_scope,
        )

    def probability(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Computes likelihoods for modules given input data.

        Likelihoods are computed from the log-likelihoods of a module.

        Args:
            data:
                Tensor containing the input data.
                Each row corresponds to a sample.
            cache:
                Optional cache dictionary.

        Returns:
            Tensor containing the likelihoods of the input data.
            Each row corresponds to an input sample.
        """
        return torch.exp(self.log_likelihood(data, cache=cache))
