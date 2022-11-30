import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure.spn import Bernoulli, BernoulliLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

        layer = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.5, 0.9])

        nodes = [
            Bernoulli(Scope([0]), p=0.2),
            Bernoulli(Scope([1]), p=0.5),
            Bernoulli(Scope([0]), p=0.9),
        ]

        dummy_data = torch.tensor([[1, 0], [0, 0], [1, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_layer_gradient_computation(self):

        p = [random.random(), random.random()]

        torch_bernoulli = BernoulliLayer(scope=[Scope([0]), Scope([1])], p=p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 2))

        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_bernoulli.p_aux.grad is not None)

        p_aux_orig = torch_bernoulli.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_bernoulli.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(p_aux_orig - torch_bernoulli.p_aux.grad, torch_bernoulli.p_aux))

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_bernoulli.p, torch_bernoulli.dist().probs))

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_bernoulli = BernoulliLayer(scope=[Scope([0]), Scope([1])], p=[0.3, 0.7])

        # create dummy data
        p_target = torch.tensor([0.8, 0.2])
        data = torch.bernoulli(p_target.unsqueeze(0).repeat((100000, 1)))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_bernoulli.parameters(), lr=0.5, momentum=0.5)

        # perform optimization (possibly overfitting)
        for i in range(50):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_bernoulli, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(torch.allclose(torch_bernoulli.p, p_target, atol=1e-2, rtol=1e-2))

    def test_likelihood_marginalization(self):

        bernoulli = BernoulliLayer(scope=[Scope([0]), Scope([1])], p=random.random())
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(bernoulli, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
