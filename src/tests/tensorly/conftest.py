from typing import Callable

import tensorly as tl
import torch
import numpy as np
import random
from pytest import fixture

@fixture(
    scope="function",
    autouse=True,
    params=["numpy", "pytorch"],
)
def do_for_all_backends(request) -> Callable:
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    backend_name = request.param
    with tl.backend_context(backend_name):
        yield backend_name