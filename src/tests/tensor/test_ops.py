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
        (T.ravel, [np.random.randn(2, 2)]),
        (T.tensor, [np.random.randn(2, 2)]),
        (T.vstack, [[np.random.randn(2, 2), np.random.randn(2, 2)]]),
        (T.hstack, [[np.random.randn(2, 2), np.random.randn(2, 2)]]),
        (T.isclose, [np.random.randn(2, 2), np.random.randn(2, 2)]),
        (T.allclose, [np.random.randn(2, 2), np.random.randn(2, 2)]),
        (T.stack, [[np.random.randn(2, 2), np.random.randn(2, 2)]]),
        (T.tolist, [np.random.randn(2, 2)]),
        (T.unique, [np.random.randint(0, 10, size=(2, 2))]),
        (T.isnan, [np.random.randn(2, 2)]),
        (T.isinf, [np.random.randn(2, 2)]),
        (T.isfinite, [np.random.randn(2, 2)]),
        (T.full, [(2, 2), 1.0]),
        (T.full, [(2, 2), 1.0]),
        (T.eigvalsh, [np.cov(np.random.randn(5, 10))]),
        (T.inv, [np.random.randn(2, 2)]),
        (T.cholesky, [np.cov(np.random.randn(2, 10))]),
        (T.svd, [np.random.randn(2, 2)]),
        (T.real, [np.random.randn(2, 2)]),
        (
            T.ix_,
            [np.random.randint(0, 10, (3,)), np.random.randint(0, 10, (3,)), np.random.randint(0, 10, (3,))],
        ),
        (T.nan_to_num, [np.random.randn(2, 2)]),
        (T.cov, [np.random.randn(2, 2), None, 1]),
        (T.repeat, [np.random.randn(2, 2), 2, 1]),
        (T.tile, [np.random.randn(2, 2), 2]),
        (T.spacing, [np.random.randn(2, 2)]),
        (T.split, [np.random.randn(2, 2), 2, 1]),
        (T.array_split, [np.random.randn(2, 2), 2, 1]),
        (T.pad_edge, [np.random.randn(2, 2), (1, 1)]),
        (T.nextafter, [np.random.randn(2, 2), np.random.randn(2, 2)]),
        (T.logsumexp, [np.random.randn(2, 2), 1]),
        (T.softmax, [np.random.randn(2, 2), 1]),
        (T.cartesian_product, [np.random.randn(2), np.random.randn(2)]),
        (T.sigmoid, [np.random.randn(2, 2)]),
        (T.ones, [2, 2]),
        (T.zeros, [2, 2]),
        (T.logical_xor, [np.random.randn(2, 2) < 0, np.random.randn(2, 2) < 0]),
        (T.logical_and, [np.random.randn(2, 2) < 0, np.random.randn(2, 2) < 0]),
        (T.pow, [np.random.randn(2, 2), 2]),
        (T.set_tensor_data, [np.random.randn(2, 2), np.random.randn(2, 2)]),
        (T.arange, [0, 1, 1]),
        (T.squeeze, [np.random.randn(2, 1, 2), 1]),
        (T.unsqueeze, [np.random.randn(2, 2), 1]),
        (T.lgamma, [np.random.rand(2, 2)]),
        (T.all, [np.random.rand(2, 2) > 0]),
        (T.any, [np.random.rand(2, 2) > 0]),
        (T.cumsum, [np.random.rand(2, 2), 0]),
        (T.cumsum, [np.random.rand(2, 2), 1]),
        (T.concatenate, [[np.random.rand(2, 2), np.random.rand(2, 2)], 0]),
        (T.concatenate, [[np.random.rand(2, 2), np.random.rand(2, 2)], 1]),
        (T.copy, [np.random.rand(2, 2)]),
        (T.exp, [np.random.rand(2, 2)]),
        (T.index_update, [np.random.rand(2, 2), (0, 0), 1]),
        (T.log, [np.random.rand(2, 2)]),
        (T.min, [np.random.rand(2, 2)]),
        (T.min, [np.random.rand(2, 2), 0]),
        (T.min, [np.random.rand(2, 2), 1]),
        (T.max, [np.random.rand(2, 2)]),
        (T.max, [np.random.rand(2, 2), 0]),
        (T.max, [np.random.rand(2, 2), 1]),
        (T.ndim, [np.random.rand(2, 2)]),
        (T.prod, [np.random.rand(2, 2), 0]),
        (T.prod, [np.random.rand(2, 2), 1]),
        (T.shape, [np.random.rand(2, 2)]),
        (T.sqrt, [np.random.rand(2, 2)]),
        (T.stack, [[np.random.rand(2, 2), np.random.rand(2, 2)], 0]),
        (T.stack, [[np.random.rand(2, 2), np.random.rand(2, 2)], 1]),
        (T.sum, [np.random.rand(2, 2), 0]),
        (T.sum, [np.random.rand(2, 2), 1]),
        (T.diag, [np.random.rand(2, 2)]),
        (T.transpose, [np.random.rand(2, 2, 2)]),
        (T.transpose, [np.random.rand(2, 2, 2), (0, 2, 1)]),
        (T.transpose, [np.random.rand(2, 2, 2), (2, 1, 0)]),
        (T.dot, [np.random.rand(2), np.random.rand(2)]),
        (T.norm, [np.random.rand(2, 2)]),
        (T.sort, [np.random.rand(10)]),
        (T.sort, [np.random.rand(10), 0]),
        (T.sort, [np.random.rand(10, 5), 1]),
        (T.where, [np.random.rand(10) > 0.5, np.random.rand(10), np.random.rand(10)]),
        (T.reshape, [np.random.rand(10), (2, 5)]),
        (T.reshape, [np.random.rand(4, 3), (2, 6)]),
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


if __name__ == "__main__":
    unittest.main()
