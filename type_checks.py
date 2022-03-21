#!/usr/bin/env python3

import numpy as np
import torch as th
from typing import Any


class OutOfBoundsException(Exception):
    def __init__(self, value, lower_bound, upper_bound):
        super().__init__(f"Value {value} was not in bounds: [{lower_bound}, {upper_bound}).")


class InvalidTypeException(Exception):
    def __init__(self, value, expected_type):
        super().__init__(
            f"Value {value} was of type {type(value)} but expected to be of type {expected_type} (or a subclass of this type) ."
        )

class InvalidStackedSpnConfigurationException(Exception):
    def __init__(self, expected, observed, parameter_name):
        super().__init__(f"The StackedSpn has received an invalid configuration: Expected {parameter_name}={expected} but got {parameter_name}={observed}.")


def _check_bounds(value: Any, expected_type, lower_bound=None, upper_bound=None):
    # Check lower bound
    if lower_bound:
        if not value >= expected_type(lower_bound):
            raise OutOfBoundsException(value, lower_bound, upper_bound)

    # Check upper bound
    if upper_bound:
        if not value < expected_type(upper_bound):
            raise OutOfBoundsException(value, lower_bound, upper_bound)


def _check_type(value: Any, expected_type):
    # Check if type is from torch
    if isinstance(value, th.Tensor):
        _check_type_torch(value, expected_type)

    # Check if type is from numpy
    elif type(value).__module__ == np.__name__:
        _check_type_numpy(value, expected_type)
    elif type(value) == int or type(value) == float:
        _check_type_core(value, expected_type)
    else:
        raise Exception(f"Unsupported type ({type(value)}) for typecheck.")


def _check_type_core(value: Any, expected_type):
    if expected_type == float and not isinstance(value, float):
        raise InvalidTypeException(value, expected_type)
    elif expected_type == int and not isinstance(value, int):
        raise InvalidTypeException(value, expected_type)
        
def _check_type_numpy(value: Any, expected_type):
    # Check float
    if expected_type == float:
        if not isinstance(value, np.floating):
            raise InvalidTypeException(value, expected_type)
    # Check integer
    elif expected_type == int:
        if not isinstance(value, np.integer):
            raise InvalidTypeException(value, expected_type)
    else:
        raise Exception(f"Unexpected data type, must be either int or float, but was {expected_type}")


def _check_type_torch(value: th.Tensor, expected_type):
    # Get torch data type
    dtype = value.dtype

    # If we expect float, check if dtype is a floating point, vice versa for int
    if expected_type == float:
        if not dtype.is_floating_point:
            raise InvalidTypeException(value, expected_type)
    elif expected_type == int:
        if dtype.is_floating_point:
            raise InvalidTypeException(value, expected_type)
    else:
        raise Exception(f"Unexpected data type, must be either int or float, but was {expected_type}")


def check_valid(value: Any, expected_type, lower_bound=None, upper_bound=None, allow_none: bool = False):
    """
    Check if a value is of a certain type and in given bounds.
    """
    if allow_none and value is None:
        return value
    if not allow_none and value is None:
        raise Exception(f"Invalid input: Got None, but expected type {expected_type}.")
    # First check if the type is valid
    _check_type(value, expected_type)

    # Then check if value is inbounds
    _check_bounds(value, expected_type, lower_bound, upper_bound)

    return expected_type(value)
