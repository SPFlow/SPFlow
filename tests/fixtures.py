import os

from typing import Callable
import random
import numpy as np
from pytest import fixture
from spflow import tensor as T

# If a backend is specified via the environment variable SPFLOW_TEST_BACKEND,
# only that backend is tested, else all backends are tested.
test_backend = os.getenv("SPFLOW_TEST_BACKEND")
if test_backend is not None and test_backend.strip() != "":
    backends = [test_backend]
else:
    backends = [T.Backend.NUMPY, T.Backend.PYTORCH, T.Backend.JAX]


@fixture(
    scope="function",
    autouse=True,
    params=backends,
)
def backend_auto(request):
    np.random.seed(0)
    random.seed(0)
    backend_name = request.param

    # TODO: What about Jax?
    if backend == T.Backend.PYTORCH:
        import torch
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
    with T.backend_context(backend_name):
        yield


@fixture(
    scope="function",
    autouse=False,
    params=backends,
)
def backend(request):
    np.random.seed(0)
    random.seed(0)
    backend = request.param

    # TODO: What about Jax?
    if backend == T.Backend.PYTORCH:
        import torch
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
    with T.backend_context(backend):
        yield backend
