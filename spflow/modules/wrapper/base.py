"""Base wrapper classes for SPFlow module adaptation.

This module provides abstract base classes and utilities for creating wrapper
modules that adapt existing SPFlow modules for specific use cases or data formats.
Wrapper modules enable the extension of SPFlow functionality without modifying
core module implementations. The wrapper pattern enables flexible extension of SPFlow capabilities while
maintaining compatibility with the core module interfaces.
"""

from abc import ABC

from spflow.meta.data import Scope
from spflow.modules.base import Module


class Wrapper(Module, ABC):
    """Abstract base class for SPFlow module wrappers.

    Provides a foundation for creating wrapper modules that adapt existing SPFlow
    modules for specific use cases, data formats, or integration scenarios. Wrapper
    modules delegate most operations to the wrapped module while providing
    specialized functionality for specific contexts. All abstract methods from Module are delegated to wrapped module,
    concrete implementations should override specific methods as needed, and wrapper modules inherit scope and structure from wrapped modules.

    The wrapper pattern enables:
    - Custom data format handling (images, sequences, etc.)
    - Preprocessing and postprocessing integration
    - External framework compatibility layers
    - Specialized input/output transformations

    Attributes:
        module (Module): The wrapped SPFlow module.
        num_repetitions (int): Number of repetitions inherited from wrapped module.
        scope (Scope): Variable scope inherited from wrapped module.
    """

    def __init__(self, module: Module):
        """Initialize wrapper with specified SPFlow module.

        Creates a wrapper that delegates most operations to wrapped module
        while allowing specialized overrides for specific functionality. The wrapped module's scope and structure are preserved,
        all abstract Module interface methods are delegated by default, and override specific methods to add custom wrapper functionality.

        Args:
            module (Module): The SPFlow module to wrap. Can be any valid
                SPFlow module including complex circuit structures.
        """
        super().__init__()
        self.module = module
        self.num_repetitions = module.num_repetitions
        self.scope = module.scope

    @property
    def out_features(self) -> int:
        """Returns the number of output features of the wrapped module.

        Delegates to the wrapped module's out_features property.

        Returns:
            int: Number of output features from the wrapped module.
        """
        return self.module.out_features

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels of the wrapped module.

        Delegates to the wrapped module's out_channels property.

        Returns:
            int: Number of output channels from the wrapped module.
        """
        return self.module.out_channels

    @property
    def feature_to_scope(self) -> list[Scope]:
        """Returns the mapping from features to scopes from the wrapped module.

        Delegates to the wrapped module's feature_to_scope property.

        Returns:
            list[Scope]: Feature-to-scope mapping from the wrapped module.
        """
        return self.module.feature_to_scope

    @property
    def device(self):
        """Returns the device of the wrapped module.

        Automatically determines the device where the wrapped module's
        parameters are located, handling multi-device scenarios gracefully.

        Returns:
            torch.device: Device where the wrapped module parameters are located.
        """
        return next(iter(self.module.parameters())).device

    def extra_repr(self) -> str:
        return f"D={self.out_features}, C={self.out_channels}, R={self.num_repetitions}"
