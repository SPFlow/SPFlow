import random
import unittest

import torch
import numpy as np
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_poisson import CondPoissonLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_poisson import CondPoisson
from spflow.torch.structure.general.layers.leaves.parametric.cond_poisson import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_no_l(self):

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, poisson, torch.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"l": [1, 1]}

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0], [2], [5]])
        targets = torch.tensor([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_l(self):

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[poisson] = {"l": [1, 1]}

        # create test inputs/outputs
        data = torch.tensor([[0], [2], [5]])
        targets = torch.tensor([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"l": torch.tensor([1, 1])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[poisson] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0], [2], [5]])
        targets = torch.tensor([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_layer_likelihood(self):

        layer = CondPoissonLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
            cond_f=lambda data: {"l": [0.2, 1.0, 2.3]},
        )

        nodes = [
            CondPoisson(Scope([0], [2]), cond_f=lambda data: {"l": 0.2}),
            CondPoisson(Scope([1], [2]), cond_f=lambda data: {"l": 1.0}),
            CondPoisson(Scope([0], [2]), cond_f=lambda data: {"l": 2.3}),
        ]

        dummy_data = torch.tensor([[1, 3], [3, 7], [2, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        l = torch.tensor(
            [random.randint(1, 10), random.randint(1, 10)],
            dtype=torch.get_default_dtype(),
            requires_grad=True,
        )

        torch_poisson = CondPoissonLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"l": l},
        )

        # create dummy input data (batch size x random variables)
        data = torch.cat([torch.randint(0, 10, (3, 1)), torch.randint(0, 10, (3, 1))], dim=1)

        log_probs_torch = log_likelihood(torch_poisson, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(l.grad is not None)

    def test_likelihood_marginalization(self):

        poisson = CondPoissonLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"l": random.random() + 1e-7},
        )
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(poisson, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        cond_f = lambda data: {"l": [1, 1]}

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0], [2], [5]])

        log_probs = log_likelihood(poisson, data)

        # make sure that probabilities match python backend probabilities
        for backend in backends:
            tl.set_backend(backend)
            layer_updated = updateBackend(poisson)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            self.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
