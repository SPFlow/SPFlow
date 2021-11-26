from functools import wraps
from typing import Any, Dict
from multipledispatch import dispatch


def memoize(key_type: Any):
    """Memoization decorator.

    Does not use its own cache, but requires the wrapped function to provide a cache dictionary.
    Additionally requires the key variable (used to look up the cache) to be the first positional arugment to the wrapped function and the cache to be passed as a keyword argument.

    Args:
        key_type:
            Type of the key variable used to look up the cache.
    """

    def memoize_decorator(f):
        @wraps(f)
        def memoized_f(*args, **kwargs):

            # get cache (initialize if non specified)
            kwargs.setdefault("cache", {})
            cache = kwargs["cache"]

            # args contains key variable to be used for cache (assumes key variable is first positional argument to f)
            if len(args) > 0:
                key = args[0]
            # key variable must be part of kwargs
            else:
                # get all key variable candidates that match the specified variable type
                key_candidates = [key for key, val in kwargs.items() if isinstance(val, key_type)]

                # no matching key variable found
                if not key_candidates:
                    raise LookupError(
                        f"Could not find argument of type {key_type} to look up cache."
                    )
                # too many matchin key variables found
                elif len(key_candidates) > 1:
                    raise LookupError(
                        f"Found not unambiguoulsy determine which argument of type {key_type} to use to look up cache."
                    )

                # get only key candidate
                key = key_candidates[0]

            if key not in cache:
                # compute result and update cache
                cache[key] = f(*args, **kwargs)

            return cache[key]

        return memoized_f

    return memoize_decorator
