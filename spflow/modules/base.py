"""Base module architecture for SPFlow probabilistic circuits.

Provides the foundational abstract base class for all SPFlow modules, defining
core interfaces for log-likelihood computation, sampling, parameter learning,
and scope management with PyTorch integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from spflow.meta.data.scope import Scope
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext


class Module(nn.Module, ABC):
    """Abstract base class for all SPFlow probabilistic circuit modules.

    Extends PyTorch's nn.Module with probabilistic circuit functionality including
    scope management, caching, and standardized interfaces for inference and learning.
    All concrete subclasses must implement the abstract methods for log-likelihood,
    sampling, and marginalization.

    Attributes:
        inputs (nn.ModuleList): List of child modules in the circuit graph.
        scope (Scope): Variable scope defining which random variables this module operates on.
    """

    def __init__(self) -> None:
        """Initialize the SPFlow module with empty input list."""
        super().__init__()
        self.inputs: nn.ModuleList = nn.ModuleList()

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Returns the number of output features of the module.

        Returns:
            int: Number of output features.
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of output channels of the module.

        Output channels represent parallel computations or multiple distributions
        over the same scope.

        Returns:
            int: Number of output channels.
        """
        pass

    @property
    def scope(self) -> Scope:
        """Variable scope defining which random variables this module operates on.

        Returns:
            Scope: The module's scope.
        """
        return self._scope

    @scope.setter
    def scope(self, scope: Scope):
        """Set the variable scope for this module.

        Args:
            scope (Scope): New variable scope.

        Raises:
            ValueError: If the scope is incompatible with module structure.
        """
        self._scope = scope

    @property
    @abstractmethod
    def feature_to_scope(self) -> list[Scope]:
        """Mapping from output features to their respective scopes.

        Returns:
            list[Scope]: List of scopes, one for each output feature.
        """
        pass

    @property
    def device(self):
        """Device where the module's parameters are located.

        Returns first parameter's device, or CPU if no parameters exist.

        Returns:
            torch.device: Device where parameters are located.
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
                    f"num_samples ({num_samples}) must match data.shape[0] ({data.shape[0]}) or be None"
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
        """Compute log likelihood P(data | module).

        Computes log probability of input data under this module's distribution.
        Uses log-space for numerical stability. Results should be cached for efficiency.

        Args:
            data (Tensor): Input data of shape (batch_size, num_features).
                NaN values indicate evidence for conditional computation.
            cache (Cache | None, optional): Cache for intermediate computations. Defaults to None.

        Returns:
            Tensor: Log-likelihood of shape (batch_size, out_features, out_channels).

        Raises:
            ValueError: If input data shape is incompatible with module scope.
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
        """Generate samples from the module's probability distribution.

        Supports both random sampling and MAP inference (via is_mpe flag).
        Handles conditional sampling through evidence in data tensor.

        Args:
            num_samples (int | None, optional): Number of samples to generate. Defaults to 1.
            data (Tensor | None, optional): Pre-allocated tensor with NaN values indicating
                where to sample. If None, creates new tensor. Defaults to None.
            is_mpe (bool, optional): If True, returns most probable values instead of
                random samples. Defaults to False.
            cache (Cache | None, optional): Cache for intermediate computations. Defaults to None.
            sampling_ctx (SamplingContext | None, optional): Context for routing samples
                through the circuit. Defaults to None.

        Returns:
            Tensor: Sampled values of shape (batch_size, num_features).

        Raises:
            ValueError: If sampling parameters are incompatible.
        """
        pass

    def sample_with_evidence(
        self,
        evidence: Tensor,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Samples from module with evidence.

        This is effectively calling log_likelihood then sampling from the module with a populated cache.

        Args:
            evidence: Evidence tensor.
            is_mpe: Boolean value indicating whether to perform maximum a posteriori estimation (MPE).
                Defaults to False.
            cache: Optional cache dictionary to reuse across calls.
            sampling_ctx: Optional sampling context containing the instances (i.e., rows) of
                ``data`` to fill with sampled values and the output indices of the node to sample from.

        Returns:
            Tensor containing the sampled values. Each row corresponds to a sample.
        """
        if cache is None:
            cache = Cache()

        self.log_likelihood(evidence, cache=cache)

        return self.sample(
            data=evidence,
            is_mpe=is_mpe,
            sampling_ctx=sampling_ctx,
            cache=cache,
        )

    def log_posterior(
            self,
            data: torch.Tensor,
            log_prior: torch.Tensor = None,
            cache: Cache | None = None,
    ) -> torch.Tensor:
        """Compute log-posterior probabilities for multi-class models.

        Args:
            data: Input data tensor.
            prior: Optional prior probabilities tensor.
            cache: Optional cache dictionary for caching intermediate results.

        Returns:
            Log-posterior probabilities.
        """
        if self.out_channels <= 1:
            raise ValueError("Posterior can only be computed for models with multiple classes.")

        if cache is None:
            cache = Cache()
        shape = (1, self.out_features, self.out_channels, *(() if self.num_repetitions is None else (self.num_repetitions,)))
        if log_prior is not None:
            assert log_prior.shape == shape, f"Expected log_prior shape {shape}, got {log_prior.shape}"
            ll_y = log_prior
        else:
            l_y = torch.ones(shape, device=self.device) / self.out_channels
            ll_y = torch.log(l_y)

        assert torch.allclose(ll_y.exp().sum(dim=2), torch.tensor(1.0)), "Prior probabilities must sum to 1 across classes."

        ll = self.log_likelihood(
            data,
            cache=cache,
        )  # shape: (batch_size, out_feature, out_channel, num_repetitions)

        # logp(y | x) = logp(x, y) - logp(x)
        #             = logp(x | y) + logp(y) - logp(x)
        #             = logp(x | y) + logp(y) - logsumexp(logp(x,y), dim=y)

        ll_x_and_y = ll + ll_y
        ll_x = torch.logsumexp(ll_x_and_y, dim=2, keepdim=True)
        ll_y_given_x = ll_x_and_y - ll_x

        return ll_y_given_x

    def expectation_maximization(
        self,
        data: Tensor,
        bias_correction: bool = True,
        cache: Cache | None = None,
    ) -> None:
        """Expectation-maximization step.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary.
        """
        if cache is None:
            cache = Cache()

        for input_module in self.inputs:
            input_module.expectation_maximization(data, cache=cache, bias_correction=bias_correction)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        bias_correction: bool = True,
        nan_strategy: str = "ignore",
        cache: Cache | None = None,
    ) -> None:
        """Update parameters via maximum likelihood estimation.

        Args:
            data: Input data tensor.
            weights: Optional sample weights.
            bias_correction: Whether to apply bias correction. Defaults to True.
            nan_strategy: Strategy for handling NaN values in data. Defaults to "ignore".
            cache: Optional cache dictionary.
        """
        if cache is None:
            cache = Cache()

        for input_module in self.inputs:
            input_module.maximum_likelihood_estimation(
                data,
                weights=weights,
                bias_correction=bias_correction,
                nan_strategy=nan_strategy,
                cache=cache,
            )

    @abstractmethod
    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        """Marginalize out specified random variables from the module.

        Computes marginal distribution by integrating out the specified variables.
        Essential for conditioning on evidence and computing marginal probabilities.

        Args:
            marg_rvs (list[int]): Random variable indices to marginalize out.
            prune (bool, optional): Whether to prune unnecessary modules during
                marginalization. Defaults to True.
            cache (Cache | None, optional): Cache for intermediate computations. Defaults to None.

        Returns:
            Module | None: Marginalized module, or None if all variables are marginalized out.

        Raises:
            ValueError: If marginalization variables are not in the module's scope.
        """
        pass

    def forward(self, data: Tensor, cache: Cache | None = None):
        """Forward pass is simply the log-likelihood function.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary.
        """
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
        """Convert this module to a readable string representation.

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
