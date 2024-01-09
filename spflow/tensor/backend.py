import logging

from contextlib import contextmanager
from enum import Enum
import numpy as np
import os

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
    def __init__(self, backend=None):
        if backend is None:
            backend = get_backend()
        super().__init__(f"Method not implemented for the backend: {backend}")


class Backend(str, Enum):
    """Enum class for the possible backends."""

    NUMPY = "numpy"
    PYTORCH = "pytorch"
    JAX = "jax"


# Global variable to store the current backend, defaults to numpy upon initialization or the value of the
# environment variable SPFLOW_BACKEND
if env := os.environ.get("SPFLOW_BACKEND"):
    _BACKEND = Backend(env)
else:
    _BACKEND = Backend.NUMPY


def get_backend():
    """Get the current backend."""
    return _BACKEND


def set_backend(backend: Backend):
    """Set the _BACKEND to the specified backend."""
    global _BACKEND
    _BACKEND = backend


def is_torch_available():
    """Check if torch is currently installed and the import was successful."""
    return IS_TORCH_AVAILABLE


def is_jax_available():
    """Check if jax is currently installed and the import was successful."""
    return IS_JAX_AVAILABLE


@contextmanager
def backend_context(backend: Backend):
    """Context manager to temporarily change the backend."""
    _old_backend = get_backend()
    set_backend(backend)
    try:
        yield
    finally:
        set_backend(_old_backend)


def is_numpy():
    """Check if the current backend is numpy."""
    return _BACKEND == Backend.NUMPY


def is_pytorch():
    """Check if the current backend is pytorch."""
    return _BACKEND == Backend.PYTORCH


def is_jax():
    """Check if the current backend is jax."""
    return _BACKEND == Backend.JAX
