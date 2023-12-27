import unittest

import numpy as np
import pytest

from spflow import tensor as T
from spflow.tensor import Tensor
from tests.fixtures import backend


def test_get_default_dtype(backend):
    assert T.get_default_dtype(1) == T.int32()
    assert T.get_default_dtype([1]) == T.int32()
    assert T.get_default_dtype(1.0) == T.float32()
    assert T.get_default_dtype([1.0]) == T.float32()


def test_isint(backend):
    assert T.isint(1) == True
    assert T.isint(1.0) == False
    assert T.isint([1, 2, 3]) == True
    assert T.isint([1.0, 2.0, 3.0]) == False
    assert T.isint(T.tensor([1, 2, 3])) == True
    assert T.isint(T.tensor([1.0, 2.0, 3.0])) == False
    assert T.isint(T.tensor([True, False, True])) == False


def test_isfloat(backend):
    assert T.isfloat(1) == False
    assert T.isfloat(1.0) == True
    assert T.isfloat([1, 2, 3]) == False
    assert T.isfloat([1.0, 2.0, 3.0]) == True
    assert T.isfloat(T.tensor([1, 2, 3])) == False
    assert T.isfloat(T.tensor([1.0, 2.0, 3.0])) == True
    assert T.isfloat(T.tensor([True, False, True])) == False


def test_isbool(backend):
    assert T.isbool(1) == False
    assert T.isbool(1.0) == False
    assert T.isbool([1, 2, 3]) == False
    assert T.isbool([1.0, 2.0, 3.0]) == False
    assert T.isbool(T.tensor([1, 2, 3])) == False
    assert T.isbool(T.tensor([1.0, 2.0, 3.0])) == False
    assert T.isbool(T.tensor([True, False, True])) == True


def test_is_tensor_dispatch_type_plum_faithful():
    """Test if the tensor type T that we dispatch on in some methods is faithful according to plum."""
    from plum import is_faithful

    # Note: if this test fails, we might get performance issues according to
    #       https://beartype.github.io/plum/types.html?highlight=performance#performance-and-faithful-types
    assert is_faithful(Tensor)


if __name__ == "__main__":
    unittest.main()
