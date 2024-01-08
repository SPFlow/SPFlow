import random
import unittest

import numpy as np
import torch
from sklearn.cross_decomposition import CCA

from spflow.torch.utils import cca

tc = unittest.TestCase()


def test_cca_1(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    # set seed
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # not symmetric adjacency matrix
    sklearn_cca = CCA(n_components=1)
    sklearn_cca.algorithm = "svd"

    data = np.vstack(
        [
            np.hstack(
                [
                    np.random.binomial(n=1, p=0.8, size=(100, 1)),
                    np.random.binomial(n=1, p=0.3, size=(100, 1)),
                ]
            ),
            np.hstack(
                [
                    np.random.binomial(n=1, p=0.3, size=(50, 1)),
                    np.random.binomial(n=1, p=0.1, size=(50, 1)),
                ]
            ),
        ]
    )
    data = data.astype(float)

    sklearn_cca.fit(data[:, [0]], data[:, [1]])

    res_sklearn = (
        sklearn_cca.x_rotations_,
        sklearn_cca.y_rotations_,
        sklearn_cca.x_weights_,
        sklearn_cca.y_weights_,
        sklearn_cca._x_scores,
        sklearn_cca._y_scores,
        sklearn_cca.x_loadings_,
        sklearn_cca.y_loadings_,
    )

    res_torch = cca(
        torch.tensor(data[:, [0]]),
        torch.tensor(data[:, [1]]),
        n_components=1,
    )

    for e_sklearn, e_torch in zip(res_sklearn, res_torch):
        tc.assertTrue(
            np.allclose(e_sklearn, e_torch.numpy(), rtol=1e-4)
            or np.allclose(e_sklearn, -e_torch.numpy(), rtol=1e-4)
        )


def test_cca_2(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return
    # set seed
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # not symmetric adjacency matrix
    sklearn_cca = CCA(n_components=2)
    sklearn_cca.algorithm = "svd"

    data = np.vstack(
        [
            np.hstack([np.random.randn(100, 2), np.random.randn(100, 2)]),
            np.hstack([np.random.randn(50, 2), np.random.randn(50, 2)]),
        ]
    )
    data = data.astype(float)

    sklearn_cca.fit(data[:, [0, 1]], data[:, [2, 3]])

    res_sklearn = (
        sklearn_cca.x_rotations_,
        sklearn_cca.y_rotations_,
        sklearn_cca.x_weights_,
        sklearn_cca.y_weights_,
        sklearn_cca._x_scores,
        sklearn_cca._y_scores,
        sklearn_cca.x_loadings_,
        sklearn_cca.y_loadings_,
    )

    res_torch = cca(
        torch.tensor(data[:, [0, 1]]),
        torch.tensor(data[:, [2, 3]]),
        n_components=2,
    )

    for i, (e_sklearn, e_torch) in enumerate(zip(res_sklearn, res_torch)):
        tc.assertTrue(
            np.allclose(e_sklearn, e_torch.numpy(), atol=1e-1, rtol=1e-1)
            or np.allclose(e_sklearn, -e_torch.numpy(), atol=1e-1, rtol=1e-1)
        )


if __name__ == "__main__":
    unittest.main()
