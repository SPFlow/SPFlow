from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.inference.nodes.leaves.parametric.gaussian import log_likelihood
from spflow.torch.inference.module import likelihood, log_likelihood
from ...structure.nodes.dummy_node import DummyNode
import torch
import unittest
import random


def create_example_spn():
    spn = SPNSumNode(
            children=[
                SPNProductNode(
                    children=[
                        Gaussian(Scope([0])),
                        SPNSumNode(
                            children=[
                                SPNProductNode(
                                    children=[
                                        Gaussian(Scope([1])),
                                        Gaussian(Scope([2])),
                                    ]
                                ),
                                SPNProductNode(
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
                SPNProductNode(
                    children=[
                        SPNProductNode(
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
    def test_likelihood(self):
        dummy_spn = create_example_spn()
        dummy_data = torch.tensor([[1.0, 0.0, 1.0]])

        l_result = likelihood(dummy_spn, dummy_data)
        ll_result = log_likelihood(dummy_spn, dummy_data)
        self.assertTrue(torch.isclose(l_result[0][0], torch.tensor(0.023358)))
        self.assertTrue(torch.isclose(ll_result[0][0], torch.tensor(-3.7568156)))

    def test_likelihood_marginalization(self):
        spn = create_example_spn()
        dummy_data = torch.tensor([[float("nan"), 0.0, 1.0]])

        l_result = likelihood(spn, dummy_data)
        ll_result = log_likelihood(spn, dummy_data)
        self.assertTrue(torch.isclose(l_result[0][0], torch.tensor(0.09653235)))
        self.assertTrue(torch.isclose(ll_result[0][0], torch.tensor(-2.33787707)))

    def test_likelihood_not_implemented(self):
        dummy_node = DummyNode()
        dummy_data = torch.tensor([[1.0]])

        self.assertRaises(NotImplementedError, log_likelihood, dummy_node, dummy_data)
        self.assertRaises(NotImplementedError, likelihood, dummy_node, dummy_data)


if __name__ == "__main__":
    unittest.main()