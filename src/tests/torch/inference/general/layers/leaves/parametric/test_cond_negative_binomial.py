import random
import unittest

import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import CondNegativeBinomial, CondNegativeBinomialLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_no_p(self):

        negative_binomial = CondNegativeBinomialLayer(
            Scope([0], [1]), n=2, n_nodes=2
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            torch.tensor([[0], [1]]),
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": [1.0, 1.0]}

        negative_binomial = CondNegativeBinomialLayer(
            Scope([0], [1]), n=2, n_nodes=2, cond_f=cond_f
        )

        # create test inputs/outputs
        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        negative_binomial = CondNegativeBinomialLayer(
            Scope([0], [1]), n=2, n_nodes=2
        )

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[negative_binomial] = {"p": [1.0, 1.0]}

        # create test inputs/outputs
        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(
            negative_binomial, data, dispatch_ctx=dispatch_ctx
        )

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        bernoulli = CondNegativeBinomialLayer(Scope([0], [1]), n=2, n_nodes=2)

        cond_f = lambda data: {"p": torch.tensor([1.0, 1.0])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_layer_likelihood(self):

        layer = CondNegativeBinomialLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
            n=[3, 2, 3],
            cond_f=lambda data: {"p": [0.2, 0.5, 0.9]},
        )

        nodes = [
            CondNegativeBinomial(
                Scope([0], [2]), n=3, cond_f=lambda data: {"p": 0.2}
            ),
            CondNegativeBinomial(
                Scope([1], [2]), n=2, cond_f=lambda data: {"p": 0.5}
            ),
            CondNegativeBinomial(
                Scope([0], [2]), n=3, cond_f=lambda data: {"p": 0.9}
            ),
        ]

        dummy_data = torch.tensor([[3, 1], [1, 2], [0, 0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat(
            [log_likelihood(node, dummy_data) for node in nodes], dim=1
        )

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        n = [random.randint(2, 10), random.randint(2, 10)]
        p = torch.tensor([random.random(), random.random()], requires_grad=True)

        torch_negative_binomial = CondNegativeBinomialLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            n=n,
            cond_f=lambda data: {"p": p},
        )

        # create dummy input data (batch size x random variables)
        data = torch.cat(
            [torch.randint(1, n[0], (3, 1)), torch.randint(1, n[1], (3, 1))],
            dim=1,
        )

        log_probs_torch = log_likelihood(torch_negative_binomial, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_negative_binomial.n.grad is None)
        self.assertTrue(p.grad is not None)

    def test_likelihood_marginalization(self):

        negative_binomial = CondNegativeBinomialLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            n=5,
            cond_f=lambda data: {"p": random.random()},
        )
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(negative_binomial, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
