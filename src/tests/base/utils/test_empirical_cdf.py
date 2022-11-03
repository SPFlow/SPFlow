from spflow.base.utils import empirical_cdf

import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_empirical_cdf(self):

        data = np.array([[0.1, 0.3], [0.5, -1.0], [0.1, 0.2], [0.0, -1.0]])

        # actual ecdf values
        target_ecdf = np.array(
            [[3 / 4, 4 / 4], [4 / 4, 2 / 4], [3 / 4, 3 / 4], [1 / 4, 2 / 4]]
        )

        ecdf = empirical_cdf(data)
        self.assertTrue(np.allclose(ecdf, target_ecdf))


if __name__ == "__main__":
    unittest.main()
