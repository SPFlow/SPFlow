from spflow.meta.data import Scope
from spflow.torch.structure.spn import Gamma, GammaLayer
from spflow.torch.inference import log_likelihood
import torch
import unittest
import random


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

        layer = GammaLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            alpha=[0.2, 1.0, 2.3],
            beta=[1.0, 0.3, 0.97],
        )

        nodes = [
            Gamma(Scope([0]), alpha=0.2, beta=1.0),
            Gamma(Scope([1]), alpha=1.0, beta=0.3),
            Gamma(Scope([0]), alpha=2.3, beta=0.97),
        ]

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat(
            [log_likelihood(node, dummy_data) for node in nodes], dim=1
        )

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        alpha = [random.randint(1, 5), random.randint(1, 3)]
        beta = [random.randint(1, 5), random.randint(2, 4)]

        torch_gamma = GammaLayer(
            scope=[Scope([0]), Scope([1])], alpha=alpha, beta=beta
        )

        # create dummy input data (batch size x random variables)
        data = torch.rand(3, 2)

        log_probs_torch = log_likelihood(torch_gamma, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_gamma.alpha_aux.grad is not None)
        self.assertTrue(torch_gamma.beta_aux.grad is not None)

        alpha_aux_orig = torch_gamma.alpha_aux.detach().clone()
        beta_aux_orig = torch_gamma.beta_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(
                alpha_aux_orig - torch_gamma.alpha_aux.grad,
                torch_gamma.alpha_aux,
            )
        )
        self.assertTrue(
            torch.allclose(
                beta_aux_orig - torch_gamma.beta_aux.grad, torch_gamma.beta_aux
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(
            torch.allclose(torch_gamma.alpha, torch_gamma.dist().concentration)
        )
        self.assertTrue(
            torch.allclose(torch_gamma.beta, torch_gamma.dist().rate)
        )

    def test_gradient_optimization(self):

        # initialize distribution
        torch_gamma = GammaLayer(
            scope=[Scope([0]), Scope([1])], alpha=[1.0, 1.2], beta=[2.0, 1.8]
        )

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.Gamma(concentration=2.0, rate=1.0).sample(
            (100000, 2)
        )

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(
            torch_gamma.parameters(), lr=0.5, momentum=0.5
        )

        # perform optimization (possibly overfitting)
        for i in range(20):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_gamma, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(
                torch_gamma.alpha, torch.tensor([2.0, 2.0]), atol=1e-3, rtol=0.3
            )
        )
        self.assertTrue(
            torch.allclose(
                torch_gamma.beta, torch.tensor([1.0, 1.0]), atol=1e-3, rtol=0.3
            )
        )

    def test_likelihood_marginalization(self):

        gamma = GammaLayer(
            scope=[Scope([0]), Scope([1])],
            alpha=random.random() + 1e-7,
            beta=random.random() + 1e-7,
        )
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(gamma, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
