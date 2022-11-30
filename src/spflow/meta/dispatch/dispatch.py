"""Contains the standard 'dispatch' decorator for SPFlow.

Typical usage example:

    @dispatch
    def foo():
        pass
"""
import warnings
from typing import Any, Callable

from plum import dispatch as plum_dispatch  # type: ignore

from spflow.meta.dispatch.memoize import memoize as memoize_decorator
from spflow.meta.dispatch.substitutable import substitutable as substitutable_decorator


def dispatch(*args, memoize=False, substitutable=True) -> Callable:
    """Decorator that wraps a function and dispatches it.

    Dispatches a function using Plum's ``dispatch`` decorator and provides additional convenient options for dispatched functions in SPFlow.

    Args:
        memoize:
            Boolean indicating whether or not the dispatched function should use memoization via the dispatch cache. Should be assigned as a keyword argument.
            If set to True, the original function is wrapped and the resulting function automatically checks the dispatch cache for an existing value for the same dispatched function name to be returned instead.
            The first argument to the original function must be an instance of (a subclass of) ``MetaModule`` to check for corresponding cached values.
            Defaults to False.
        substitutable:
            Boolean indicating whether or not the dispatched function can be substituted by a different function via the dispatch cache. Should be assigned as a keyword argument.
            If set to True, the original function is wrapped and the resulting function automatically checks the dispatch cache for an alternative specified function to be called instead.
            The first argument to the original function must be an instance of (a subclass of) ``MetaModule`` to check for corresponding alternative functions.
            Defaults to True.

    Raises:
        ValueError: Invalid arguments received.

    Returns:
        Wrapped function that is dispatched based on its function signature.
    """

    def dispatch_decorator(f):
        _f = f

        if substitutable:
            _f = substitutable_decorator(_f)
        if memoize:
            _f = memoize_decorator(_f)

        if f.__name__ in ["log_likelihood"] and not memoize:
            warnings.warn(
                f"Function '{f.__name__}' was dispatched without memoization, but is recommended to save redundant computations and to correctly compute gradients in certain backends. Ignore this message if this is intended."
            )
        elif (
            f.__name__ in ["em", "maximum_likelihood_estimation"]
            and not memoize
        ):
            warnings.warn(
                f"Function '{f.__name__}' was dispatched without memoization, but is recommended to not optimize the same module twice in one pass. Ignore this message if this is intended."
            )
        elif f.__name__ in ["toBase", "toTorch", "marginalize"] and not memoize:
            warnings.warn(
                f"Function '{f.__name__}' was dispatched without memoization, but is recommended to not handle the same module twice in one pass and maintain correct structure. Ignore this message if this is intended."
            )

        return plum_dispatch(_f)

    if len(args) == 1 and callable(args[0]):
        f = args[0]
        return dispatch_decorator(f)
    elif len(args) > 0:
        raise ValueError(
            "'dispatch' Decorator received unknown positional arguments. Try keyword arguments instead."
        )
    else:
        return dispatch_decorator
