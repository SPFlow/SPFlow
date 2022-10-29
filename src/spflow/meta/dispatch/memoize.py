# -*- coding: utf-8 -*-
"""Contains a convenient decorator to automatically check for cached values during dispatch.

Typical usage example:

    @memoize
    def foo():
        pass
"""
from typing import Callable, Any
from functools import wraps
from spflow.meta.structure.module import MetaModule
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    default_dispatch_context,
)


def memoize(f) -> Callable:
    """Cecorator that wraps a function and automatically checks for cached values during dispatching.

    Wraps a function in order to automatically check the dispatch cache for stored values.
    The first argument to the original function must be an instance of (a subclass of) ``MetaModule`` to use as a key for the cache.

    Returns:
        Wrapped function that automatically checks against a dispatch cache
    """

    @wraps(f)
    def memoized_f(*args, **kwargs) -> Any:

        # ----- retrieve MetaModule that is dispatched on -----

        # first argument is the key
        key = args[0]

        if not isinstance(key, MetaModule):
            raise ValueError(
                f"First argument is expected to be of type {MetaModule}, but was of type {type(key)}."
            )

        # ----- retrieve DispatchContext -----

        # look in kwargs
        if "dispatch_ctx" in kwargs:
            dispatch_ctx = kwargs["dispatch_ctx"]
        # look in args
        else:
            dispatch_ctx = None
            _args = []

            for arg in args:
                if isinstance(arg, DispatchContext):

                    # check if another dispatch context has been found before
                    if dispatch_ctx is not None:
                        raise LookupError(
                            f"Multiple positional candidates of type {DispatchContext} found. Cannot determine which one to use."
                        )

                    dispatch_ctx = kwargs["dispatch_ctx"] = arg
                else:
                    # append argument to list of non-DispatchContext arguments
                    _args.append(arg)

            # replace args with argument list without dispatch context (makes retrieving it easier in subsequent recursions)
            args = _args

        # check if dispatch context is 'None'
        if dispatch_ctx is None:
            dispatch_ctx = default_dispatch_context()

        # ----- memoization -----

        if f.__name__ not in dispatch_ctx.cache:
            dispatch_ctx.cache[f.__name__] = {}

        f_cache = dispatch_ctx.cache[f.__name__]

        if key not in f_cache:
            # compute result and update cache
            f_cache[key] = f(*args, **kwargs)

        return f_cache[key]

    return memoized_f
