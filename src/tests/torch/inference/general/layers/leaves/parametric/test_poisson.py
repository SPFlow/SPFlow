import random
import unittest

import torch

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure.spn import Poisson, PoissonLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

        layer = PoissonLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3]
        )

        nodes = [
            Poisson(Scope([0]), l=0.2),
            Poisson(Scope([1]), l=1.0),
            Poisson(Scope([0]), l=2.3),
        ]

        dummy_data = torch.tensor([[1, 3], [3, 7], [2, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat(
            [log_likelihood(node, dummy_data) for node in nodes], dim=1
        )

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        l = torch.tensor([random.randint(1, 10), random.randint(1, 10)])

        torch_poisson = PoissonLayer(scope=[Scope([0]), Scope([1])], l=l)

        # create dummy input data (batch size x random variables)
        data = torch.cat(
            [torch.randint(0, 10, (3, 1)), torch.randint(0, 10, (3, 1))], dim=1
        )

        log_probs_torch = log_likelihood(torch_poisson, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_poisson.l_aux.grad is not None)

        l_aux_orig = torch_poisson.l_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(
                l_aux_orig - torch_poisson.l_aux.grad, torch_poisson.l_aux
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(
            torch.allclose(torch_poisson.l, torch_poisson.dist().rate)
        )

    def test_gradient_optimization(self):

        # initialize distribution
        torch_poisson = PoissonLayer(scope=[Scope([0]), Scope([1])], l=1.0)

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.Poisson(rate=4.0).sample((100000, 2))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=0.1)

        # perform optimization (possibly overfitting)
        for i in range(40):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_poisson, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(
                torch_poisson.l, torch.tensor([4.0, 4.0]), atol=1e-3, rtol=0.3
            )
        )

    def test_likelihood_marginalization(self):

        poisson = PoissonLayer(
            scope=[Scope([0]), Scope([1])], l=random.random() + 1e-7
        )
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(poisson, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
