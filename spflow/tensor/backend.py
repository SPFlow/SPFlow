import logging

from contextlib import contextmanager
from enum import Enum
import numpy as np

from typing import Union

logger = logging.getLogger(__name__)

# List of supported tensor types
_TENSOR_TYPES = [np.ndarray]
try:
    import torch

    logger.debug("PyTorch backend loaded.")
    IS_TORCH_AVAILABLE = True

    _TENSOR_TYPES.append(torch.Tensor)
except ImportError as e:
    IS_TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    logger.info("Jax backend loaded.")

    from jax.lib import xla_bridge

    _jax_platform = xla_bridge.get_backend().platform
    logger.debug("Jax backend platform: %s", _jax_platform)

    _TENSOR_TYPES.append(jnp.ndarray)
    IS_JAX_AVAILABLE = True
except ImportError as e:
    IS_JAX_AVAILABLE = False

# Define tensor type as union of all supported tensor types s.t. dispatch methods can be defined for all of them
_TENSOR_TYPES = tuple(_TENSOR_TYPES)
Tensor = Union[_TENSOR_TYPES]


class MethodNotImplementedError(NotImplementedError):
    def __init__(self, backend):
        super().__init__(f"Method not implemented for the backend: {backend}")


class Backend(str, Enum):
    """Enum class for the possible backends."""

    NUMPY = "numpy"
    PYTORCH = "pytorch"
    JAX = "jax"


# Global variable to store the current backend, defaults to numpy upon initialization
_BACKEND = Backend.NUMPY


def get_backend():
    """Get the current backend."""
    return _BACKEND


def set_backend(backend: Backend):
    """Set the _BACKEND to the specified backend."""
    global _BACKEND
    _BACKEND = backend


def is_torch_available():
    return IS_TORCH_AVAILABLE


def is_jax_available():
    return IS_JAX_AVAILABLE


@contextmanager
def backend_context(backend: Backend):
    _old_backend = get_backend()
    set_backend(backend)
    try:
        yield
    finally:
        set_backend(_old_backend)
