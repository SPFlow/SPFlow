from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure.spn import CondGeometric, CondGeometricLayer
from spflow.torch.inference import log_likelihood, likelihood
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

    def test_likelihood_no_p(self):

        geometric = CondGeometricLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(
            ValueError, log_likelihood, geometric, torch.tensor([[0], [1]])
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": [0.2, 0.5]}

        geometric = CondGeometricLayer(
            Scope([0], [1]), n_nodes=2, cond_f=cond_f
        )

        # create test inputs/outputs
        data = torch.tensor([[1], [5], [10]])
        targets = torch.tensor(
            [[0.2, 0.5], [0.08192, 0.03125], [0.0268435, 0.000976563]]
        )

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        geometric = CondGeometricLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[geometric] = {"p": [0.2, 0.5]}

        # create test inputs/outputs
        data = torch.tensor([[1], [5], [10]])
        targets = torch.tensor(
            [[0.2, 0.5], [0.08192, 0.03125], [0.0268435, 0.000976563]]
        )

        probs = likelihood(geometric, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(geometric, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        geometric = CondGeometricLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"p": torch.tensor([0.2, 0.5])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[geometric] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[1], [5], [10]])
        targets = torch.tensor(
            [[0.2, 0.5], [0.08192, 0.03125], [0.0268435, 0.000976563]]
        )

        probs = likelihood(geometric, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(geometric, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_layer_likelihood(self):

        layer = CondGeometricLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
            cond_f=lambda data: {"p": [0.2, 1.0, 0.3]},
        )

        nodes = [
            CondGeometric(Scope([0], [2]), cond_f=lambda data: {"p": 0.2}),
            CondGeometric(Scope([1], [2]), cond_f=lambda data: {"p": 1.0}),
            CondGeometric(Scope([0], [2]), cond_f=lambda data: {"p": 0.3}),
        ]

        dummy_data = torch.tensor([[4, 1], [3, 7], [1, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat(
            [log_likelihood(node, dummy_data) for node in nodes], dim=1
        )

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        p = torch.tensor([random.random(), random.random()], requires_grad=True)

        torch_geometric = CondGeometricLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"p": p},
        )

        # create dummy input data (batch size x random variables)
        data = torch.randint(1, 10, (3, 2))

        log_probs_torch = log_likelihood(torch_geometric, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(p.grad is not None)

    def test_likelihood_marginalization(self):

        geometric = CondGeometricLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"p": random.random() + 1e-7},
        )
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