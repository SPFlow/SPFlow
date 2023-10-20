import random
import unittest

import torch

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure.spn import Exponential, ExponentialLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

        layer = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])

        nodes = [
            Exponential(Scope([0]), l=0.2),
            Exponential(Scope([1]), l=1.0),
            Exponential(Scope([0]), l=2.3),
        ]

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        l = [random.random(), random.random()]

        torch_exponential = ExponentialLayer(scope=[Scope([0]), Scope([1])], l=l)

        # create dummy input data (batch size x random variables)
        data = torch.rand(3, 2)

        log_probs_torch = log_likelihood(torch_exponential, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_exponential.l_aux.grad is not None)

        l_aux_orig = torch_exponential.l_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(
                l_aux_orig - torch_exponential.l_aux.grad,
                torch_exponential.l_aux,
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_exponential.l, torch_exponential.dist().rate))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_exponential = ExponentialLayer(scope=[Scope([0]), Scope([1])], l=[0.5, 0.7])

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.Exponential(rate=1.5).sample((100000, 2))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=0.5)

        # perform optimization (possibly overfitting)
        for i in range(20):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_exponential, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(
                torch_exponential.l,
                torch.tensor([1.5, 1.5]),
                atol=1e-3,
                rtol=0.3,
            )
        )

    def test_likelihood_marginalization(self):

        exponential = ExponentialLayer(scope=[Scope([0]), Scope([1])], l=random.random() + 1e-7)
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(exponential, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()