from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.binomial import BinomialLayer
from spflow.torch.inference.layers.leaves.parametric.binomial import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.binomial import Binomial
from spflow.torch.inference.nodes.leaves.parametric.binomial import log_likelihood
from spflow.torch.inference.module import log_likelihood
import torch
import numpy as np
import unittest
import itertools
import random


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

        layer = BinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], n=[3, 2, 3], p=[0.2, 0.5, 0.9])

        nodes = [
            Binomial(Scope([0]), n=3, p=0.2),
            Binomial(Scope([1]), n=2, p=0.5),
            Binomial(Scope([0]), n=3, p=0.9),
        ]

        dummy_data = torch.tensor([[3, 1], [1, 2], [0, 0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        n = [4, 6]
        p = random.random()

        torch_binomial = BinomialLayer(scope=[Scope([0]), Scope([1])], n=n, p=p)

        # create dummy input data (batch size x random variables)
        data = torch.tensor([[0, 5], [3, 2], [4, 1]])

        log_probs_torch = log_likelihood(torch_binomial, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_binomial.n.grad is None)
        self.assertTrue(torch_binomial.p_aux.grad is not None)

        n_orig = torch_binomial.n.detach().clone()
        p_aux_orig = torch_binomial.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_binomial.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(n_orig, torch_binomial.n))
        self.assertTrue(
            torch.allclose(p_aux_orig - torch_binomial.p_aux.grad, torch_binomial.p_aux)
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.equal(torch_binomial.n, torch_binomial.dist().total_count.long()))
        self.assertTrue(torch.allclose(torch_binomial.p, torch_binomial.dist().probs))

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_binomial = BinomialLayer(scope=[Scope([0]), Scope([1])], n=[5, 3], p=0.3)

        # create dummy data
        p_target = torch.tensor([0.8, 0.5])
        data = torch.distributions.Binomial(torch.tensor([5, 3]), p_target).sample((100000,))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_binomial.parameters(), lr=0.5)

        # perform optimization (possibly overfitting)
        for i in range(50):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_binomial, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(torch_binomial.p, p_target, atol=1e-3, rtol=1e-3)
        )

    def test_likelihood_marginalization(self):

        binomial = BinomialLayer(scope=[Scope([0]), Scope([1])], n=5, p=random.random())
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(binomial, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))
    
    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()