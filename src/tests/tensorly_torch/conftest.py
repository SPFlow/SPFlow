from typing import Callable

import tensorly as tl
from pytest import fixture


CONFIG_VALUE = "pytorch"



@fixture(
    scope="function", # function?
    autouse=True,
    params=["numpy", "pytorch"], # test only numpy
)
def do_for_all_backends(request) -> Callable:
    backend_name = request.param
    with tl.backend_context(backend_name):
        yield backend_name  # we would not have to yield anything useful, could also be None