"""
Created on August 03, 2022

@authors: Philipp Deibert
"""
from typing import Callable, Any
from functools import wraps
from plum import dispatch as plum_dispatch
from spflow.meta.dispatch.memoize import memoize as memoize_decorator
from spflow.meta.dispatch.swappable import swappable as swappable_decorator


def dispatch(*args, memoize=False, swappable=True) -> Callable:
    """TODO."""
    def dispatch_decorator(f):
        _f = f

        if(swappable): _f = swappable_decorator(_f)
        if(memoize): _f = memoize_decorator(_f)

        return plum_dispatch(_f) 

    if(len(args) == 1 and callable(args[0])):
        f = args[0]
        return dispatch_decorator(f)
    elif(len(args) > 0):
        raise ValueError("'dipatch' decorator received unknown positional arguments. Try keyword arguments instead.")
    else:
        return dispatch_decorator