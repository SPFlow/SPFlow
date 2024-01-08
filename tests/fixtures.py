import os

from typing import Callable
import random
import numpy as np
import torch
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
    if T.Backend.PYTORCH in backends:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
    # TODO: What about Jax?
    np.random.seed(0)
    random.seed(0)
    backend_name = request.param
    with T.backend_context(backend_name):
        yield


@fixture(
    scope="function",
    autouse=False,
    params=backends,
)
def backend(request):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    backend_name = request.param
    with T.backend_context(backend_name):
        yield backend_name
