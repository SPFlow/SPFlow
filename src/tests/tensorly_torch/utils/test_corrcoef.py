import random
import unittest

import numpy as np
import torch

from spflow.torch.utils import corrcoef


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_corrcoef(self):

        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        data = np.hstack(
            [
                np.random.multivariate_normal(np.array([-1.0, 3.0]), np.eye(2), (100,)),
                np.random.randn(100, 1),
            ]
        )

        corrcoef_np = np.corrcoef(data[:, [0, 1]].T, data[:, [2]].T)
        corrcoef_torch = corrcoef(torch.tensor(data))

        self.assertTrue(np.allclose(corrcoef_np, corrcoef_torch.numpy()))


if __name__ == "__main__":
    unittest.main()