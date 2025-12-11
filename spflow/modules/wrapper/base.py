"""Base wrapper classes for SPFlow module adaptation.

This module provides abstract base classes and utilities for creating wrapper
modules that adapt existing SPFlow modules for specific use cases or data formats.
Wrapper modules enable the extension of SPFlow functionality without modifying
core module implementations. The wrapper pattern enables flexible extension of SPFlow capabilities while
maintaining compatibility with the core module interfaces.
"""

from abc import ABC
import numpy as np

from spflow.meta.data import Scope
from spflow.modules.module import Module


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
        self.scope = module.scope

        # Shape computation: delegate to wrapped module
        self.in_shape = self.module.in_shape
        self.out_shape = self.module.out_shape

    @property
    def feature_to_scope(self) -> np.ndarray:
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
        """Return a string representation of the wrapper module.

        Provides a concise representation showing the output features (D),
        output channels (C), and number of repetitions (R) for debugging
        and logging purposes.

        Returns:
            str: String representation in format "D={out_features}, C={out_channels}, R={num_repetitions}".
        """
        return f"D={self.out_shape.features}, C={self.out_shape.channels}, R={self.out_shape.repetitions}"
