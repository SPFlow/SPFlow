import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import CondCategorical, CondCategoricalLayer


class TestCondCategorical(unittest.TestCase):

    def test_likelihood_no_p(self):

        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, layer, torch.tensor([[0], [1]]))


    def test_likelihood_module_cond_f(self):

        k = [2, 2]
        p = [[0.5, 0.5], [0.3, 0.7]]
        cond_f = lambda data: {"k": k, "p": p}

        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.5, 0.3], [0.5, 0.7]])

        probs = likelihood(layer, data)
        log_probs = log_likelihood(layer, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    
    def test_likelihood_args_p(self):

        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[layer] = {"k": [2, 2], "p": [[0.5, 0.5], [0.3, 0.7]]}

        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.5, 0.3], [0.5, 0.7]])

        probs = likelihood(layer, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))


    def test_likelihood_args_cond_f(self):

        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=2)

        k = [2, 2]
        p = [[0.5, 0.5], [0.3, 0.7]]
        cond_f = lambda data: {"k": k, "p": p}


        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[layer] = {"cond_f": cond_f}

        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.5, 0.3], [0.5, 0.7]])

        probs = likelihood(layer, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))


    def test_layer_likelihood(self):

        scopes = [Scope([0], [2]), Scope([1], [2]), Scope([0], [2])]
        ks = [2, 2, 2]
        ps = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]

        layer = CondCategoricalLayer(
            scope=scopes,
            cond_f = lambda data: {"k": ks, "p": ps}
        )

        nodes = [
            CondCategorical(scope=scopes[0], cond_f=lambda data: {"k": ks[0], "p": ps[0]}), 
            CondCategorical(scope=scopes[1], cond_f=lambda data: {"k": ks[1], "p": ps[1]}), 
            CondCategorical(scope=scopes[2], cond_f=lambda data: {"k": ks[2], "p": ps[2]})
        ]

        data = torch.tensor([[1, 0], [0, 0], [1, 1]])

        layer_probs = log_likelihood(layer, data)
        nodes_probs = torch.concat([log_likelihood(node, data) for node in nodes], axis=1)

        print(layer_probs)
        print(nodes_probs)

        self.assertTrue(torch.allclose(layer_probs, nodes_probs))


    def test_layer_gradient_computation(self):

        k = [2, 2]
        p = torch.tensor([[0.5, 0.5], [0.3, 0.7]], requires_grad=True)

        torch_categorical = CondCategoricalLayer(scope=[Scope([0], [2]), Scope([1], [2])], cond_f = lambda data: {"k": k, "p": p})

        data = np.random.randint(0., k, (10, 2))
        
        log_probs_torch = log_likelihood(torch_categorical, torch.tensor(data))

        targets_torch = torch.ones(10, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(p.grad is not None)


    def test_likelihood_marginalization(self):

        layer = CondCategoricalLayer(
            scope=[Scope([0], [2]), Scope([1], [2])], 
            cond_f=lambda data: {"k": 2, "p": [0.3, 0.7]}
        )
        data = torch.tensor([[float("nan"), float("nan")]])

        probs = log_likelihood(layer, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()