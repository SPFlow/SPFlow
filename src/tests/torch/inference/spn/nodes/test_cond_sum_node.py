import unittest

import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import CondSumNode, Gaussian, ProductNode


def create_example_spn():
    spn = CondSumNode(
        children=[
            ProductNode(
                children=[
                    Gaussian(Scope([0])),
                    CondSumNode(
                        children=[
                            ProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                            ProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                        ],
                        cond_f=lambda data: {
                            "weights": torch.tensor([0.3, 0.7])
                        },
                    ),
                ],
            ),
            ProductNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0])),
                            Gaussian(Scope([1])),
                        ]
                    ),
                    Gaussian(Scope([2])),
                ]
            ),
        ],
        cond_f=lambda data: {"weights": torch.tensor([0.4, 0.6])},
    )
    return spn


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood(self):
        dummy_spn = create_example_spn()
        dummy_data = torch.tensor([[1.0, 0.0, 1.0]])

        l_result = likelihood(dummy_spn, dummy_data)
        ll_result = log_likelihood(dummy_spn, dummy_data)
        self.assertTrue(torch.isclose(l_result[0][0], torch.tensor(0.023358)))
        self.assertTrue(
            torch.isclose(ll_result[0][0], torch.tensor(-3.7568156))
        )

    def test_likelihood_marginalization(self):
        spn = create_example_spn()
        dummy_data = torch.tensor([[float("nan"), 0.0, 1.0]])

        l_result = likelihood(spn, dummy_data)
        ll_result = log_likelihood(spn, dummy_data)
        self.assertAlmostEqual(l_result[0][0], 0.09653235)
        self.assertAlmostEqual(ll_result[0][0], -2.33787707)

    def test_likelihood_marginalization(self):
        spn = create_example_spn()
        dummy_data = torch.tensor([[float("nan"), 0.0, 1.0]])

        l_result = likelihood(spn, dummy_data)
        ll_result = log_likelihood(spn, dummy_data)
        self.assertTrue(torch.isclose(l_result[0][0], torch.tensor(0.09653235)))
        self.assertTrue(
            torch.isclose(ll_result[0][0], torch.tensor(-2.33787707))
        )

    def test_sum_node_gradient_computation(self):

        torch.manual_seed(0)

        # generate random weights for a sum node with two children
        weights = torch.tensor([0.3, 0.7], requires_grad=True)

        data_1 = torch.randn((70000, 1))
        data_1 = (data_1 - data_1.mean()) / data_1.std() + 5.0
        data_2 = torch.randn((30000, 1))
        data_2 = (data_2 - data_2.mean()) / data_2.std() - 5.0

        data = torch.cat([data_1, data_2])

        # initialize Gaussians
        gaussian_1 = Gaussian(Scope([0]), 5.0, 1.0)
        gaussian_2 = Gaussian(Scope([0]), -5.0, 1.0)

        # sum node to be optimized
        sum_node = CondSumNode(
            children=[gaussian_1, gaussian_2],
            cond_f=lambda data: {"weights": weights},
        )

        ll = log_likelihood(sum_node, data).mean()
        ll.backward()

        self.assertTrue(weights.grad is not None)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
