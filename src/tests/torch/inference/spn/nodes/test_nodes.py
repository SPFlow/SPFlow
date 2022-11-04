from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure.spn import SumNode, ProductNode, Gaussian
from spflow.torch.utils.projections import (
    proj_convex_to_real,
    proj_real_to_convex,
)
from spflow.torch.inference import log_likelihood, likelihood
from ....structure.general.nodes.dummy_node import DummyNode
import torch
import unittest


def create_example_spn():
    spn = SumNode(
        children=[
            ProductNode(
                children=[
                    Gaussian(Scope([0])),
                    SumNode(
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
                        weights=torch.tensor([0.3, 0.7]),
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
        weights=torch.tensor([0.4, 0.6]),
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
        self.assertTrue(torch.isclose(l_result[0][0], torch.tensor(0.09653235)))
        self.assertTrue(
            torch.isclose(ll_result[0][0], torch.tensor(-2.33787707))
        )

    def test_dummy_node_likelihood_not_implemented(self):
        dummy_node = DummyNode()
        dummy_data = torch.tensor([[1.0]])

        self.assertRaises(
            NotImplementedError, log_likelihood, dummy_node, dummy_data
        )
        self.assertRaises(
            NotImplementedError, likelihood, dummy_node, dummy_data
        )

    def test_sum_node_gradient_optimization(self):

        torch.manual_seed(0)

        # generate random weights for a sum node with two children
        weights = torch.tensor([0.3, 0.7])

        data_1 = torch.randn((70000, 1))
        data_1 = (data_1 - data_1.mean()) / data_1.std() + 5.0
        data_2 = torch.randn((30000, 1))
        data_2 = (data_2 - data_2.mean()) / data_2.std() - 5.0

        data = torch.cat([data_1, data_2])

        # initialize Gaussians
        gaussian_1 = Gaussian(Scope([0]), 5.0, 1.0)
        gaussian_2 = Gaussian(Scope([0]), -5.0, 1.0)

        # freeze Gaussians
        gaussian_1.requires_grad = False
        gaussian_2.requires_grad = False

        # sum node to be optimized
        sum_node = SumNode(
            children=[gaussian_1, gaussian_2],
            weights=weights,
        )

        # make sure that weights are correctly projected
        self.assertTrue(torch.allclose(weights, sum_node.weights))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(sum_node.parameters(), lr=0.5)

        for i in range(50):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log likelihood
            nll = -log_likelihood(sum_node, data).mean()
            nll.backward()

            if i == 0:
                # check a few general things (just for the first update)

                # check if gradients are computed
                self.assertTrue(sum_node.weights_aux.grad is not None)

                # update parameters
                optimizer.step()

                # verify that sum node weights are still valid after update
                self.assertTrue(
                    torch.isclose(sum_node.weights.sum(), torch.tensor(1.0))
                )
            else:
                # update parameters
                optimizer.step()

        self.assertTrue(
            torch.allclose(
                sum_node.weights, torch.tensor([0.7, 0.3]), atol=1e-3, rtol=1e-3
            )
        )

    def test_projection(self):

        self.assertTrue(
            torch.allclose(
                proj_real_to_convex(torch.randn(5)).sum(), torch.tensor(1.0)
            )
        )

        weights = torch.rand(5)
        weights /= weights.sum()

        self.assertTrue(
            torch.allclose(proj_convex_to_real(weights), torch.log(weights))
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
