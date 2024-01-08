import random
from unittest import SkipTest
from typing import Callable

import numpy as np
import tensorly as tl
from pytest import fixture

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow
except ImportError:
    tensorflow = None

try:
    import jax
except ImportError:
    jax = None


def make_all_deterministic(seed: int = 42) -> None:
    """Set seeds and other things to ensure reproducibility."""
    # Python's built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    if tensorflow is not None:
        tensorflow.random.set_seed(seed)
        tensorflow.config.experimental.enable_op_determinism()

    # Apparently, JAX does not have a global seed, so we cannot set it here.


@fixture(
    scope="function",
    autouse=True,
    params=["numpy", "pytorch", "tensorflow", "jax"],
)
def do_for_all_backends(request) -> Callable:
    """Runs a test for all backends while ensuring reproducibility.

    Tests for which this fixture is used should have a ``do_for_all_backends`` argument.
    This suffices to run the test for all backends.

    If a backend is not installed, the test is makred skipped.
    """
    make_all_deterministic()

    backend_name = request.param

    if backend_name == "pytorch" and torch is None:
        raise SkipTest("PyTorch is not installed")

    if backend_name == "tensorflow" and tensorflow is None:
        raise SkipTest("TensorFlow is not installed")

    if backend_name == "jax" and jax is None:
        raise SkipTest("JAX is not installed")

    with tl.backend_context(backend_name):
        yield backend_name
