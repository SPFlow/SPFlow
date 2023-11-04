from typing import Callable

import tensorly as tl
from pytest import fixture

@fixture(
    scope="function",
    autouse=True,
    params=["numpy", "pytorch"],
)
def do_for_all_backends(request) -> Callable:
    backend_name = request.param
    with tl.backend_context(backend_name):
        yield backend_name