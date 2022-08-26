from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.log_normal import LogNormalLayer
from spflow.torch.inference.layers.leaves.parametric.log_normal import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.log_normal import LogNormal
from spflow.torch.inference.nodes.leaves.parametric.log_normal import log_likelihood
from spflow.torch.inference.module import log_likelihood
import torch
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

        layer = LogNormalLayer(scope=[Scope([0]), Scope([1]), Scope([0])], mean=[0.2, 1.0, 2.3], std=[1.0, 0.3, 0.97])

        nodes = [
            LogNormal(Scope([0]), mean=0.2, std=1.0),
            LogNormal(Scope([1]), mean=1.0, std=0.3),
            LogNormal(Scope([0]), mean=2.3, std=0.97),
        ]

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        mean = [random.random(), random.random()]
        std = [random.random() + 1e-8, random.random() + 1e-8]  # offset by small number to avoid zero

        torch_log_normal = LogNormalLayer(scope=[Scope([0]), Scope([1])], mean=mean, std=std)

        # create dummy input data (batch size x random variables)
        data = torch.rand(3, 2)

        log_probs_torch = log_likelihood(torch_log_normal, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_log_normal.mean.grad is not None)
        self.assertTrue(torch_log_normal.std_aux.grad is not None)

        mean_orig = torch_log_normal.mean.detach().clone()
        std_aux_orig = torch_log_normal.std_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(mean_orig - torch_log_normal.mean.grad, torch_log_normal.mean)
        )
        self.assertTrue(
            torch.allclose(
                std_aux_orig - torch_log_normal.std_aux.grad, torch_log_normal.std_aux
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_log_normal.mean, torch_log_normal.dist().loc))
        self.assertTrue(torch.allclose(torch_log_normal.std, torch_log_normal.dist().scale))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_log_normal = LogNormalLayer(scope=[Scope([0]), Scope([1])], mean=[1.1, 0.8], std=[2.0, 1.5])

        torch.manual_seed(10)

        # create dummy data
        data = torch.distributions.LogNormal(0.0, 1.0).sample((100000, 2))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=0.5, momentum=0.5)

        # perform optimization (possibly overfitting)
        for i in range(20):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_log_normal, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(torch_log_normal.mean, torch.tensor([0.0, 0.0]), atol=1e-2, rtol=0.2)
        )
        self.assertTrue(
            torch.allclose(torch_log_normal.std, torch.tensor([1.0, 1.0]), atol=1e-2, rtol=0.2)
        )

    def test_likelihood_marginalization(self):
        
        log_normal = LogNormalLayer(scope=[Scope([0]), Scope([1])], mean=random.random(), std=random.random()+1e-7)
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(log_normal, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()