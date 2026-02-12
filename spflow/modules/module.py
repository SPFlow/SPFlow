"""Base module architecture for SPFlow probabilistic circuits.

Provides the foundational abstract base class for all SPFlow modules, defining
core interfaces for log-likelihood computation, sampling, parameter learning,
and scope management with PyTorch integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from spflow.meta.data.scope import Scope
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import (
    SamplingContext,
    require_sampling_context,
)
from spflow.modules.module_shape import ModuleShape


class Module(nn.Module, ABC):
    """Abstract base class for all SPFlow probabilistic circuit modules.

    Extends PyTorch's nn.Module with probabilistic circuit functionality including
    scope management, caching, and standardized interfaces for inference and learning.
    All concrete subclasses must implement the abstract methods for log-likelihood,
    sampling, and marginalization.

    Attributes:
        inputs (Module | None): Child module in the circuit graph. None for leaf modules.
        scope (Scope): Variable scope defining which random variables this module operates on.
    """

    def __init__(self) -> None:
        """Initialize the module with no input."""
        super().__init__()
        # Shape attributes - should be set by subclass __init__
        self._in_shape: ModuleShape = None
        self._out_shape: ModuleShape = None

    @property
    def inputs(self) -> Module | Iterable[Module]:
        """Returns the input module, or None for leaf modules.

        Returns:
            Module | None: The child input module, or None if this is a leaf module.
        """
        return self._modules.get("inputs", None)

    @inputs.setter
    def inputs(self, value: Module) -> None:
        """Set the input module.

        Args:
            value: The module to set as input.
        """
        self._modules["inputs"] = value

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
    def feature_to_scope(self) -> np.ndarray:
        """Mapping from output features to their respective scopes.

        Returns:
            np.ndarray[Scope]: 2D-array of scopes. Each row corresponds to an output feature,
                each column to a repetition.
        """
        pass

    @property
    def in_shape(self) -> ModuleShape:
        """Expected input tensor shape (features, channels, repetitions).

        For leaf modules, returns the shape of data tensors: (features, 1, 1).

        Returns:
            ModuleShape: The expected input shape.
        """
        return self._in_shape

    @in_shape.setter
    def in_shape(self, value: ModuleShape) -> None:
        """Set the input shape.

        Args:
            value: The ModuleShape to set as input shape.
        """
        self._in_shape = value

    @property
    def out_shape(self) -> ModuleShape:
        """Output tensor shape (features, channels, repetitions).

        Returns:
            ModuleShape: The output shape produced by this module.
        """
        return self._out_shape

    @out_shape.setter
    def out_shape(self, value: ModuleShape) -> None:
        """Set the output shape.

        Args:
            value: The ModuleShape to set as output shape.
        """
        self._out_shape = value

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

    def _prepare_internal_sampling_inputs(
        self,
        data: Tensor | None,
        sampling_ctx: SamplingContext | None,
    ) -> tuple[Tensor, SamplingContext]:
        """Require a prepared data tensor and strict sampling context for internal modules."""
        if data is None:
            raise ValueError("Internal _sample(...) requires a prepared data tensor.")
        sampling_ctx = require_sampling_context(
            sampling_ctx,
            num_samples=data.shape[0],
            module_out_shape=self.out_shape,
            device=data.device,
        )
        return data, sampling_ctx

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
                NaN values indicate missing values to marginalize over.
            cache (Cache | None, optional): Cache for intermediate computations. Defaults to None.

        Returns:
            Tensor: Log-likelihood of shape (batch_size, out_features, out_channels).

        Raises:
            ValueError: If input data shape is incompatible with module scope.
        """
        pass

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
        data = self._prepare_sample_data(num_samples=num_samples, data=data)
        if cache is None:
            cache = Cache()
        sampling_ctx = require_sampling_context(
            sampling_ctx,
            num_samples=data.shape[0],
            module_out_shape=self.out_shape,
            device=data.device,
        )
        return self._sample(
            data=data,
            sampling_ctx=sampling_ctx,
            cache=cache,
            is_mpe=is_mpe,
        )

    @abstractmethod
    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        """Internal sampling hook used for recursive sampling calls.

        This method assumes that root-level sampling preparation has already been
        handled by ``sample(...)``. Internal module recursion must call
        ``_sample(...)``.
        """
        pass

    def mpe(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate most probable explanation from the module's probability distribution.

        This is a convenience method that calls sample with is_mpe=True.

        Handles conditional sampling through evidence in data tensor.

        Args:
            num_samples (int | None, optional): Number of samples to generate. Defaults to 1.
            data (Tensor | None, optional): Pre-allocated tensor with NaN values indicating
                where to sample. If None, creates new tensor. Defaults to None.
            cache (Cache | None, optional): Cache for intermediate computations. Defaults to None.
            sampling_ctx (SamplingContext | None, optional): Context for routing samples
                through the circuit. Defaults to None.

        Returns:
            Tensor: MPE values of shape (batch_size, num_features).

        Raises:
            ValueError: If sampling parameters are incompatible.
        """
        return self.sample(
            num_samples=num_samples,
            data=data,
            is_mpe=True,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

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

    def __input_is_a_list(self):
        ok = False
        if hasattr(self, "inputs") and self.inputs is not None:
            inputs = self.inputs
            if (
                hasattr(inputs, "__iter__")
                and not isinstance(inputs, (tuple, list))
                and inputs.__class__.__name__ == "ModuleList"
            ):
                ok = True
            elif isinstance(inputs, list):
                ok = True
        return ok

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        """Protected EM-step hook used by global EM training loop.

        Subclasses with learnable parameters should override this method.
        For composite modules without learnable parameters, the default behavior
        delegates recursively to input modules.

        Args:
            data: Input data tensor.
            bias_correction: Whether to apply bias correction. Defaults to True.
            cache: Cache dictionary populated by a preceding forward pass.
        """
        if self.__input_is_a_list():
            for child in self.inputs:
                child._expectation_maximization_step(data, cache=cache, bias_correction=bias_correction)
        elif self.inputs is not None:
            self.inputs._expectation_maximization_step(data, cache=cache, bias_correction=bias_correction)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must override _expectation_maximization_step "
                "or provide input modules that implement it."
            )

    @abstractmethod
    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        """Structurally marginalize out specified random variables from the module.

        Computes a new module representing the marginal distribution by integrating out
        the specified variables from the structure. For data-level marginalization,
        use NaNs in ``log_likelihood`` inputs.

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
        return f"D={self.out_shape.features}, C={self.out_shape.channels}, R={self.out_shape.repetitions}"

    def _extra_vis_info(self) -> str | None:
        """Return extra visualization information for this module.

        This hook method allows subclasses to provide custom visualization info
        that will be appended to the module's visualization label. Return None
        (the default) for no extra info, or a string with additional lines.

        The returned string should be formatted for monospace font display.
        Multi-line strings should use newline characters to separate lines.

        Returns:
            str | None: Extra visualization info, or None for no extra info.

        Examples:
            >>> class MyModule(Module):
            ...     def _extra_vis_info(self):
            ...         return "Custom: value"
        """
        return None

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
            Sum [D=2, C=3] [weights: (2, 2, 3, 1)] → scope: 0-1
            └─ Normal [D=2, C=2] → scope: 0-1
            >>> print(model.to_str(format="pytorch"))  # PyTorch format
            Sum(
              D=2, C=3, R=1, weights=(2, 2, 3, 1)
              (inputs): Normal(D=2, C=2, R=1)
            )
            >>> print(model.to_str(max_depth=2))  # Limit depth
            Sum [D=2, C=3] [weights: (2, 2, 3, 1)] → scope: 0-1
            └─ Normal [D=2, C=2] → scope: 0-1
        """
        from spflow.utils.module_display import module_to_str

        return module_to_str(
            self,
            format=format,
            max_depth=max_depth,
            show_params=show_params,
            show_scope=show_scope,
        )

    def print_structure_stats(self) -> str:
        """Return a readable text overview of this module's structure statistics.

        This is intended for quick debugging/logging in experiments and mirrors the
        traversal behavior used by ``to_str()`` (skipping internal ``Cat`` and
        ``ModuleList`` wrappers).

        Returns:
            Multi-line string summary of structure statistics.
        """
        from spflow.utils.structure_stats import get_structure_stats, structure_stats_to_str

        return structure_stats_to_str(get_structure_stats(self))

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
