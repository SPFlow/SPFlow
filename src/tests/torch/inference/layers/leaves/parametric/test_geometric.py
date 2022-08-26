from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.geometric import GeometricLayer
from spflow.torch.inference.layers.leaves.parametric.geometric import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.torch.inference.nodes.leaves.parametric.geometric import log_likelihood
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

        layer = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 1.0, 0.3])

        nodes = [
            Geometric(Scope([0]), p=0.2),
            Geometric(Scope([1]), p=1.0),
            Geometric(Scope([0]), p=0.3),
        ]

        dummy_data = torch.tensor([[4, 1], [3, 7], [1, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        p = [random.random(), random.random()]

        torch_geometric = GeometricLayer(scope=[Scope([0]), Scope([1])], p=p)

        # create dummy input data (batch size x random variables)
        data = torch.randint(1, 10, (3, 2))

        log_probs_torch = log_likelihood(torch_geometric, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_geometric.p_aux.grad is not None)

        p_aux_orig = torch_geometric.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(p_aux_orig - torch_geometric.p_aux.grad, torch_geometric.p_aux)
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_geometric.p, torch_geometric.dist().probs))

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_geometric = GeometricLayer(scope=[Scope([0]), Scope([1])], p=[0.3, 0.4])

        # create dummy data
        p_target = 0.8
        data = torch.distributions.Geometric(p_target).sample((100000, 2)) + 1

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=0.9, momentum=0.6)

        # perform optimization (possibly overfitting)
        for i in range(40):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_geometric, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(torch_geometric.p, torch.tensor(p_target), atol=1e-3, rtol=1e-3)
        )

    def test_likelihood_marginalization(self):
        
        geometric = GeometricLayer(scope=[Scope([0]), Scope([1])], p=random.random()+1e-7)
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(geometric, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))


    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()