import unittest

import numpy as np
import pytest

import spflow.tensor.dtype
from spflow import tensor as T
from spflow.tensor import Tensor
from tests.fixtures import backend


# The parametrization below looks at all relevant spflow.tensor.ops functions and tests them with some
# data input for each backend and ensures that the results are the same.
# A touple in the list below has the form (operation, args) where operation is the function to test and args are the
# arguments to pass to the function. The function is then called as operation(*args) and the results are compared.
@pytest.mark.parametrize(
    "operation, args",
    [
        (T.ravel, [T.randn(2, 2)]),
        (T.tensor, [T.randn(2, 2)]),
        (T.vstack, [[T.randn(2, 2), T.randn(2, 2)]]),
        (T.hstack, [[T.randn(2, 2), T.randn(2, 2)]]),
        (T.isclose, [T.randn(2, 2), T.randn(2, 2)]),
        (T.allclose, [T.randn(2, 2), T.randn(2, 2)]),
        (T.stack, [[T.randn(2, 2), T.randn(2, 2)]]),
        (T.tolist, [T.randn(2, 2)]),
        (T.unique, [T.randint(0, 10, size=(2, 2))]),
        (T.isnan, [T.randn(2, 2)]),
        (T.isinf, [T.randn(2, 2)]),
        (T.isfinite, [T.randn(2, 2)]),
        (T.full, [(2, 2), 1.0]),
        (T.full, [(2, 2), 1.0]),
        (T.eigvalsh, [T.cov(T.randn(5, 10))]),
        (T.inv, [T.randn(2, 2)]),
        (T.cholesky, [T.cov(T.randn(2, 10))]),
        (T.real, [T.randn(2, 2)]),
        (T.ix_, [T.randint(0, 10, (3,)), T.randint(0, 10, (3,)), T.randint(0, 10, (3,))]),
        (T.nan_to_num, [T.randn(2, 2)]),
        (T.cov, [T.randn(2, 2), None, 1]),
        (T.repeat, [T.randn(2, 2), 2, 1]),
        (T.tile, [T.randn(2, 2), 2]),
        (T.spacing, [T.randn(2, 2)]),
        (T.split, [T.randn(2, 2), 2, 1]),
        (T.array_split, [T.randn(2, 2), 2, 1]),
        (T.pad_edge, [T.randn(2, 2), (1, 1)]),
        (T.nextafter, [T.randn(2, 2), T.randn(2, 2)]),
        (T.logsumexp, [T.randn(2, 2), 1]),
        (T.softmax, [T.randn(2, 2), 1]),
        (T.cartesian_product, [T.randn(2), T.randn(2)]),
        (T.sigmoid, [T.randn(2, 2)]),
        (T.ones, [2, 2]),
        (T.zeros, [2, 2]),
        (T.logical_xor, [T.randn(2, 2) < 0, T.randn(2, 2) < 0]),
        (T.logical_and, [T.randn(2, 2) < 0, T.randn(2, 2) < 0]),
        (T.pow, [T.randn(2, 2), 2]),
        (T.set_tensor_data, [T.randn(2, 2), T.randn(2, 2)]),
        (T.arange, [0, 1, 1]),
        (T.squeeze, [T.randn(2, 1, 2), 1]),
        (T.unsqueeze, [T.randn(2, 2), 1]),
        (T.lgamma, [T.rand(2, 2)]),
        (T.all, [T.rand(2, 2) > 0]),
        (T.any, [T.rand(2, 2) > 0]),
        (T.cumsum, [T.rand(2, 2), 0]),
        (T.cumsum, [T.rand(2, 2), 1]),
        (T.concatenate, [[T.rand(2, 2), T.rand(2, 2)], 0]),
        (T.concatenate, [[T.rand(2, 2), T.rand(2, 2)], 1]),
        (T.copy, [T.rand(2, 2)]),
        (T.exp, [T.rand(2, 2)]),
        (T.index_update, [T.rand(2, 2), (0, 0), 1]),
        (T.log, [T.rand(2, 2)]),
        (T.min, [T.rand(2, 2)]),
        (T.min, [T.rand(2, 2), 0]),
        (T.min, [T.rand(2, 2), 1]),
        (T.max, [T.rand(2, 2)]),
        (T.max, [T.rand(2, 2), 0]),
        (T.max, [T.rand(2, 2), 1]),
        (T.abs, [T.randn(2, 2)]),
        (T.ndim, [T.rand(2, 2)]),
        (T.prod, [T.rand(2, 2), 0]),
        (T.prod, [T.rand(2, 2), 1]),
        (T.shape, [T.rand(2, 2)]),
        (T.sqrt, [T.rand(2, 2)]),
        (T.stack, [[T.rand(2, 2), T.rand(2, 2)], 0]),
        (T.stack, [[T.rand(2, 2), T.rand(2, 2)], 1]),
        (T.sum, [T.rand(2, 2), 0]),
        (T.sum, [T.rand(2, 2), 1]),
        (T.diag, [T.rand(2, 2)]),
        (T.transpose, [T.rand(2, 2, 2)]),
        (T.transpose, [T.rand(2, 2, 2), (0, 2, 1)]),
        (T.transpose, [T.rand(2, 2, 2), (2, 1, 0)]),
        (T.dot, [T.rand(2), T.rand(2)]),
        (T.norm, [T.rand(2, 2)]),
        (T.sort, [T.rand(10)]),
        (T.sort, [T.rand(10), 0]),
        (T.sort, [T.rand(10, 5), 1]),
        (T.where, [T.rand(10) > 0.5, T.rand(10), T.rand(10)]),
        (T.reshape, [T.rand(10), (2, 5)]),
        (T.reshape, [T.rand(4, 3), (2, 6)]),
        # Add more operations with their arguments here
    ],
)
def test_tensor_ops(operation, args):
    results = []
    backends = [T.Backend.JAX, T.Backend.PYTORCH, T.Backend.NUMPY]
    for i, backend in enumerate(backends):
        with T.backend_context(backend):
            try:
                result = operation(*args)

                # If the result is a list or tuple, convert all elements to numpy arrays
                if isinstance(result, (list, tuple)):
                    tensor = [np.asarray(r) for r in result]
                else:
                    # Represent as numpy array for all comparisons
                    tensor = np.asarray(result)
            except Exception as e:
                e.args = (f"[{backend}] {e.args[0]}", *e.args[1:])
                raise e

            if len(results) == 0:
                # First result, nothing to compare against
                results.append(tensor)
            else:
                # Following results, compare against previous result
                tensor_prev = results[-1]

                if not isinstance(tensor, list):
                    tensor = [tensor]
                    tensor_prev = [tensor_prev]

                for t_i, t_prev_i in zip(tensor, tensor_prev):
                    assert (
                        t_prev_i.shape == t_i.shape
                    ), f"Shape mismatch ({backends[i-1]}: {t_prev_i.shape}, {backend}: {t_i.shape}) with args {args}."
                    assert (
                        t_prev_i.dtype == t_i.dtype
                    ), f"Dtype mismatch ({backends[i-1]}: {t_prev_i.dtype}, {backend}: {t_i.dtype}) with args {args}."
                    assert np.allclose(
                        t_prev_i, t_i, atol=1e-6
                    ), f"Value mismatch for args {args} between backend {backend} and {backends[i-1]}"


def test_svd():
    """
    Test svd separately since U and VH are not unique up to signs of the column vectors.
    Therefore, we have to compare the absolute values of U and VH.
    """
    data = T.randn(2, 2)

    def fn(data):
        u, s, vh = T.svd(data)
        return T.abs(u), T.abs(s), T.abs(vh)

    test_tensor_ops(operation=fn, args=(data,))


if __name__ == "__main__":
    unittest.main()
