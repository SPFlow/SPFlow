from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.cond_gamma import (
    CondGammaLayer,
)
from spflow.torch.inference.layers.leaves.parametric.cond_gamma import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_gamma import CondGamma
from spflow.torch.inference.nodes.leaves.parametric.cond_gamma import (
    log_likelihood,
)
from spflow.torch.inference.module import log_likelihood, likelihood
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

    def test_likelihood_no_alpha(self):

        gamma = CondGammaLayer(
            Scope([0]), cond_f=lambda data: {"beta": [1.0, 1.0]}, n_nodes=2
        )
        self.assertRaises(
            KeyError, log_likelihood, gamma, torch.tensor([[0], [1]])
        )

    def test_likelihood_no_beta(self):

        gamma = CondGammaLayer(
            Scope([0]), cond_f=lambda data: {"alpha": [1.0, 1.0]}, n_nodes=2
        )
        self.assertRaises(
            KeyError, log_likelihood, gamma, torch.tensor([[0], [1]])
        )

    def test_likelihood_no_alpha_beta(self):

        gamma = CondGammaLayer(Scope([0]), n_nodes=2)
        self.assertRaises(
            ValueError, log_likelihood, gamma, torch.tensor([[0], [1]])
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

        gamma = CondGammaLayer(Scope([0]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
        targets = torch.tensor(
            [[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]]
        )

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args(self):

        gamma = CondGammaLayer(Scope([0]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gamma] = {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

        # create test inputs/outputs
        data = torch.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
        targets = torch.tensor(
            [[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]]
        )

        probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        gamma = CondGammaLayer(Scope([0]), n_nodes=2)

        cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gamma] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
        targets = torch.tensor(
            [[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]]
        )

        probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_layer_likelihood(self):

        layer = CondGammaLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            cond_f=lambda data: {
                "alpha": [0.2, 1.0, 2.3],
                "beta": [1.0, 0.3, 0.97],
            },
        )

        nodes = [
            CondGamma(
                Scope([0]), cond_f=lambda data: {"alpha": 0.2, "beta": 1.0}
            ),
            CondGamma(
                Scope([1]), cond_f=lambda data: {"alpha": 1.0, "beta": 0.3}
            ),
            CondGamma(
                Scope([0]), cond_f=lambda data: {"alpha": 2.3, "beta": 0.97}
            ),
        ]

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat(
            [log_likelihood(node, dummy_data) for node in nodes], dim=1
        )

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        alpha = torch.tensor(
            [random.randint(1, 5), random.randint(1, 3)],
            dtype=torch.get_default_dtype(),
            requires_grad=True,
        )
        beta = torch.tensor(
            [random.randint(1, 5), random.randint(2, 4)],
            dtype=torch.get_default_dtype(),
            requires_grad=True,
        )

        torch_gamma = CondGammaLayer(
            scope=[Scope([0]), Scope([1])],
            cond_f=lambda data: {"alpha": alpha, "beta": beta},
        )

        # create dummy input data (batch size x random variables)
        data = torch.rand(3, 2)

        log_probs_torch = log_likelihood(torch_gamma, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(alpha.grad is not None)
        self.assertTrue(beta.grad is not None)

    def test_likelihood_marginalization(self):

        gamma = CondGammaLayer(
            scope=[Scope([0]), Scope([1])],
            cond_f=lambda data: {
                "alpha": random.random() + 1e-7,
                "beta": random.random() + 1e-7,
            },
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
