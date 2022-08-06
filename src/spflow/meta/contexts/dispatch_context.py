"""
Created on August 02, 2022

@authors: Philipp Deibert
"""
from typing import Union


class DispatchContext():
    """Class storing context information for dispatched functions.

    Stores context information in dispatched functions. Is supposed to be passed along recursive calls.

    Attributes:
        args: dictionary mapping module instances to dictionaries of additional arguments or other information.
        funcs: dictionary mapping module types to callables (may be used to specify alternative to originally dispatched functions).
        cache: dictionary mapping module instances to backend-dependent data containers (can be used for memoization).
    """
    def __init__(self):
        """TODO."""
        self.args = {}
        self.funcs = {}
        self.cache = {}


def default_dispatch_context() -> DispatchContext:
    return DispatchContext()


def init_default_dispatch_context(dispatch_ctx: Union[DispatchContext, None]) -> DispatchContext:
    return dispatch_ctx if dispatch_ctx else default_dispatch_context()