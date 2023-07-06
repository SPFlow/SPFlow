import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_bernoulli import CondBernoulliLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_bernoulli import CondBernoulli


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_no_p(self):

        bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, bernoulli, torch.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": [0.8, 0.5]}

        bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"p": [0.8, 0.5]}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"p": torch.tensor([0.8, 0.5])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_layer_likelihood(self):

        layer = CondBernoulliLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
            cond_f=lambda data: {"p": [0.2, 0.5, 0.9]},
        )

        nodes = [
            CondBernoulli(Scope([0], [2]), cond_f=lambda data: {"p": 0.2}),
            CondBernoulli(Scope([1], [2]), cond_f=lambda data: {"p": 0.5}),
            CondBernoulli(Scope([0], [2]), cond_f=lambda data: {"p": 0.9}),
        ]

        dummy_data = torch.tensor([[1, 0], [0, 0], [1, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_layer_gradient_computation(self):

        p = torch.tensor([random.random(), random.random()], requires_grad=True)

        torch_bernoulli = CondBernoulliLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"p": p},
        )

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 2))

        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(p.grad is not None)

    def test_likelihood_marginalization(self):

        bernoulli = CondBernoulliLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"p": random.random()},
        )
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
