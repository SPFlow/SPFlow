"""
Testcase for PyTorch computation backend
"""
import unittest
import numpy as np
from torch import cuda
from spn.gpu.PyTorch import Param
from spn.gpu.PyTorch import Network

parameters = Param.Param()


class TestPytorch(unittest.TestCase):
    def test_binary_values(self):
        net = Network.LayerwiseSPN(is_cuda=cuda.is_available())

        leaves = net.add_binary_nodes(2)

        sum1 = net.add_sum_nodes(3)

        weights1 = np.array([[2, 8, 0, 0], [1, 9, 0, 0], [0, 0, 4, 6]], dtype="float32").T

        net.add_sum_edges(lower_level=leaves, upper_level=sum1, weights=weights1, parameters=parameters)

        prod1 = net.add_product_nodes(2)
        mask1 = np.array([[1, 0], [0, 1], [1, 1]])
        net.add_product_edges(sum1, prod1, connections=mask1)

        sum_final = net.add_sum_nodes(1)
        weights_final = np.array([[0.3, 0.7]], dtype="float32").T
        net.add_sum_edges(lower_level=prod1, upper_level=sum_final, weights=weights_final, parameters=parameters)

        p = net.compute_unnormalized({leaves: np.array([[0, 1], [1, 0]], dtype="float32")})
        z = net.compute_unnormalized({leaves: np.ones((2, 2)).astype("float32")})
        self.assertTrue(np.isclose(p, 34.8))
        self.assertTrue(np.isclose(z, 100.00001))


if __name__ == "__main__":
    unittest.main()
