import random
import unittest

import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import CondLogNormal, CondLogNormalLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_no_mean(self):

        log_normal = CondLogNormalLayer(
            Scope([0], [1]),
            cond_f=lambda data: {"std": [0.25, 0.25]},
            n_nodes=2,
        )
        self.assertRaises(
            KeyError, log_likelihood, log_normal, torch.tensor([[0], [1]])
        )

    def test_likelihood_no_std(self):

        log_normal = CondLogNormalLayer(
            Scope([0], [1]), cond_f=lambda data: {"mean": [0.0, 0.0]}, n_nodes=2
        )
        self.assertRaises(
            KeyError, log_likelihood, log_normal, torch.tensor([[0], [1]])
        )

    def test_likelihood_no_mean_std(self):

        log_normal = CondLogNormalLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(
            ValueError, log_likelihood, log_normal, torch.tensor([[0], [1]])
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"mean": [0.0, 0.0], "std": [0.25, 0.25]}

        log_normal = CondLogNormalLayer(
            Scope([0], [1]), n_nodes=2, cond_f=cond_f
        )

        # create test inputs/outputs
        data = torch.tensor([[0.5], [1.0], [1.5]])
        targets = torch.tensor([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args(self):

        log_normal = CondLogNormalLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[log_normal] = {
            "mean": [0.0, 0.0],
            "std": [0.25, 0.25],
        }

        # create test inputs/outputs
        data = torch.tensor([[0.5], [1.0], [1.5]])
        targets = torch.tensor([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        log_normal = CondLogNormalLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"mean": [0.0, 0.0], "std": [0.25, 0.25]}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[log_normal] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0.5], [1.0], [1.5]])
        targets = torch.tensor([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_layer_likelihood(self):

        layer = CondLogNormalLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
            cond_f=lambda data: {
                "mean": [0.2, 1.0, 2.3],
                "std": [1.0, 0.3, 0.97],
            },
        )

        nodes = [
            CondLogNormal(
                Scope([0], [2]), cond_f=lambda data: {"mean": 0.2, "std": 1.0}
            ),
            CondLogNormal(
                Scope([1], [2]), cond_f=lambda data: {"mean": 1.0, "std": 0.3}
            ),
            CondLogNormal(
                Scope([0], [2]), cond_f=lambda data: {"mean": 2.3, "std": 0.97}
            ),
        ]

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat(
            [log_likelihood(node, dummy_data) for node in nodes], dim=1
        )

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        mean = torch.tensor(
            [random.random(), random.random()], requires_grad=True
        )
        std = torch.tensor(
            [random.random() + 1e-8, random.random() + 1e-8], requires_grad=True
        )  # offset by small number to avoid zero

        torch_log_normal = CondLogNormalLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"mean": mean, "std": std},
        )

        # create dummy input data (batch size x random variables)
        data = torch.rand(3, 2)

        log_probs_torch = log_likelihood(torch_log_normal, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(mean.grad is not None)
        self.assertTrue(std.grad is not None)

    def test_likelihood_marginalization(self):

        log_normal = CondLogNormalLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {
                "mean": random.random(),
                "std": random.random() + 1e-7,
            },
        )
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
