import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.utils import empirical_cdf

tc = unittest.TestCase()


def test_empirical_cdf(do_for_all_backends):
    data = tl.tensor([[0.1, 0.3], [0.5, -1.0], [0.1, 0.2], [0.0, -1.0]])

    # actual ecdf values
    target_ecdf = tl.tensor([[3 / 4, 4 / 4], [4 / 4, 2 / 4], [3 / 4, 3 / 4], [1 / 4, 2 / 4]])

    ecdf = empirical_cdf(data)
    tc.assertTrue(np.allclose(ecdf, target_ecdf))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
