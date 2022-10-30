# -*- coding: utf-8 -*-
"""Contains the dispatch context used in SPFlow

Typical usage example:

    dispatch_ctx = DispatchContext()
"""
from spflow.meta.structure.module import MetaModule
from typing import Any, Union, Dict, Any


class DispatchContext:
    """Class for storing context information for dispatched functions.

    Stores context information on dispatched functions. Is supposed to be passed along recursive calls.
    Can provide additional or required arguments for modules, specify alternative functions to be called instead of the original
    dispatched functions, as wall as store cached values.

    Attributes:
        args:
            Dictionary mapping module instances to dictionaries of additional arguments or other information.
        funcs:
            Dictionary mapping module types to callables (may be used to specify alternatives to the originally dispatched functions).
        cache:
            Dictionary mapping module instances to objects (can be used for memoization).
    """

    def __init__(self) -> None:
        """Initializes 'DispatchContext' object."""
        self.args = {}
        self.funcs = {}
        self.cache = {}

    def cache_value(
        self, f_name: str, key: MetaModule, value: Any, overwrite=True
    ) -> None:
        """Caches an object for a given function name and key.

        Args:
            f_name:
                String denoting the function for which to cache the value.
            key:
                Instance of (a subclass of) ``MetaModule`` for which to cache the value.
            value:
                Object to cache.
            overwrite:
                Boolean specifying whether or not to overwrite any potentially existing cached object.
                Defaults to True.
        """
        if not overwrite:
            # if value is already cached don't overwrite
            if self.is_cached(f_name, key):
                return

        # store value
        self.cache[f_name][key] = value

    def is_cached(self, f_name: str, key: MetaModule) -> bool:
        """Checks if a value is cached for a given function name and key.

        Args:
            f_name:
                String denoting the function for which to check the cache.
            key:
                Instance of (a subclass of) ``MetaModule`` for which to check the cache.

        Returns:
            Boolean indicating whether there is a corresponding cached object (True) or not (False).
        """
        return f_name in self.cache and key in self.cache[f_name]

    def get_cached_value(
        self, f_name: str, key: MetaModule
    ) -> Union[Any, None]:
        """Tries to retrieve a cached object for a given function name and key.

        Args:
            f_name:
                String denoting the function for which to retrive the cached object.
            key:
                Instance of (a subclass of) ``MetaModule`` for which to retrieve the cached object.

        Returns:
            Cached object (if it exists) or None.
        """
        # if no value is cached, return None
        if not self.is_cached(f_name, key):
            return None

        # return cached value
        return self.cache[f_name][key]

    def update_args(self, key: MetaModule, kwargs: Dict[str, Any]) -> None:
        """Updates additional kword arguments for a given key.

        Args:
            key:
                Instance of (a subclass of) ``MetaModule`` for which to update additional keyword arguments.
            update_args:
                Dictionary mapping strings (i.e., keyword arguments) to objects (i.e., argument values).
        """
        # create empty dictionary if no argument dictionary exists
        if key not in self.args:
            self.args[key] = {}

        # update argument dictionary
        self.args[key].update(kwargs)


def default_dispatch_context() -> DispatchContext:
    """Returns empty ``DispatchContext`` object.

    Returns:
        Empty dispatch context.
    """
    return DispatchContext()


def init_default_dispatch_context(
    dispatch_ctx: Union[DispatchContext, None]
) -> DispatchContext:
    """Initializes dispatch context, if it is not already initialized.

    Args
        dispatch_ctx:
            ``DispatchContext`` object or None.

    Returns:
        Original dispatch context if not None or a new empty dispatch context.
    """
    return (
        dispatch_ctx if dispatch_ctx is not None else default_dispatch_context()
    )
