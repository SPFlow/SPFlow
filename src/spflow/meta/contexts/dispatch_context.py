"""
Created on August 02, 2022

@authors: Philipp Deibert
"""
from spflow.meta.structure.module import MetaModule
from typing import Any, Union, Dict, Any


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
    
    def cache_value(self, f_name: str, key: MetaModule, value: Any, overwrite=True) -> None:

        if(not overwrite):
            # if value is already cached don't overwrite
            if(self.is_cached(f_name, key)):
                return

        # store value
        self.cache[f_name][key] = value

    def is_cached(self, f_name: str, key: MetaModule) -> bool:
        return (f_name in self.cache and key in self.cache[f_name])

    def get_cache_value(self, f_name: str, key: MetaModule) -> Union[Any, None]:

        # if no value is cached, return None
        if(not self.is_cached(f_name, key)):
            return None
        
        # return cached value
        return self.cache[f_name][key]
    
    def update_args(self, module: MetaModule, kwargs: Dict[str, Any]) -> None:

        # create empty dictionary if no argument dictionary exists
        if module not in self.args:
            self.args[module] = {}
        
        # update argument dictionary
        self.args[module].update(kwargs)


def default_dispatch_context() -> DispatchContext:
    return DispatchContext()


def init_default_dispatch_context(dispatch_ctx: Union[DispatchContext, None]) -> DispatchContext:
    return dispatch_ctx if dispatch_ctx is not None else default_dispatch_context()