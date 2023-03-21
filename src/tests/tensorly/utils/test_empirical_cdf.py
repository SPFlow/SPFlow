import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.utils import empirical_cdf


class TestNode(unittest.TestCase):
    def test_empirical_cdf(self):

        data = tl.tensor([[0.1, 0.3], [0.5, -1.0], [0.1, 0.2], [0.0, -1.0]])

        # actual ecdf values
        target_ecdf = tl.tensor([[3 / 4, 4 / 4], [4 / 4, 2 / 4], [3 / 4, 3 / 4], [1 / 4, 2 / 4]])

        ecdf = empirical_cdf(data)
        self.assertTrue(tl_allclose(ecdf, target_ecdf))


if __name__ == "__main__":
    unittest.main()
