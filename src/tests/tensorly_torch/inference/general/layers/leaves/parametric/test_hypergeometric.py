import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_hypergeometric import HypergeometricLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_hypergeometric import Hypergeometric
from spflow.torch.structure.general.layers.leaves.parametric.hypergeometric import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

        layer = HypergeometricLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            N=[5, 7, 5],
            M=[2, 5, 2],
            n=[3, 2, 3],
        )

        nodes = [
            Hypergeometric(Scope([0]), N=5, M=2, n=3),
            Hypergeometric(Scope([1]), N=7, M=5, n=2),
            Hypergeometric(Scope([0]), N=5, M=2, n=3),
        ]

        dummy_data = torch.tensor([[2, 1], [0, 2], [1, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=N, M=M, n=n)

        # create dummy input data (batch size x random variables)
        data = torch.tensor([[5, 6], [10, 5]])

        log_probs_torch = log_likelihood(torch_hypergeometric, data)

        # create dummy targets
        targets_torch = torch.ones(2, 2)
        targets_torch.requires_grad = True

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_hypergeometric.N.grad is None)
        self.assertTrue(torch_hypergeometric.M.grad is None)
        self.assertTrue(torch_hypergeometric.n.grad is None)

        # make sure distribution has no (learnable) parameters
        #self.assertFalse(list(torch_hypergeometric.parameters()))

    def test_likelihood_marginalization(self):

        hypergeometric = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=5, M=3, n=4)
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(hypergeometric, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        layer = HypergeometricLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            N=[5, 7, 5],
            M=[2, 5, 2],
            n=[3, 2, 3],
        )
        dummy_data = torch.tensor([[2, 1], [0, 2], [1, 1]])

        layer_ll = log_likelihood(layer, dummy_data)

        # make sure that probabilities match python backend probabilities
        for backend in backends:
            tl.set_backend(backend)
            layer_updated = updateBackend(layer)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            # check conversion from torch to python
            self.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(log_probs_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()