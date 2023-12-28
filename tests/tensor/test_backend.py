import spflow.tensor.dtype
from spflow import tensor as T
from spflow.tensor import Tensor
import unittest
import numpy as np
import pytest

from tests.fixtures import backend


def test_set_backend():
    for backend in T.Backend:
        T.set_backend(backend)
        assert T.get_backend() == backend


def test_istensor(backend):
    assert T.istensor(1) == False
    assert T.istensor(1.0) == False
    assert T.istensor([1, 2, 3]) == False
    assert T.istensor([1.0, 2.0, 3.0]) == False
    assert T.istensor(T.tensor([1, 2, 3])) == True
    assert T.istensor(T.tensor([1.0, 2.0, 3.0])) == True
    assert T.istensor(T.tensor([True, False, True])) == True


if __name__ == "__main__":
    unittest.main()
