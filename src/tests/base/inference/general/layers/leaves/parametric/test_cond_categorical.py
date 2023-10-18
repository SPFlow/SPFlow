import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import CondCategorical, CondCategoricalLayer, ProductNode, SumNode
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestCondCategorical(unittest.TestCase):
    def test_likelihood_no_p(self):

        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, layer, np.array([[0], [1]]))


    def test_likelihood_module_cond_f(self):
        cond_f = lambda data: {"k": [2, 2], "p": [[0.5, 0.5], [0.3, 0.7]]}

        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[0.5, 0.3], [0.5, 0.7]])

        probs = likelihood(layer, data)
        log_probs = log_likelihood(layer, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    
    def test_likelihood_args_p(self):

        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[layer] = {"k": [2, 2], "p": [[0.5, 0.5], [0.3, 0.7]]}

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[0.5, 0.3], [0.5, 0.7]])

        probs = likelihood(layer, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))


    def test_likelihood_args_cond_f(self):

        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"k": [2, 2], "p": [[0.5, 0.5], [0.3, 0.7]]}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[layer] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[0.5, 0.3], [0.5, 0.7]])

        probs = likelihood(layer, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))


    def test_layer_likelihood_1(self):

        layer = CondCategoricalLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"k": [2, 2], "p": [[0.5, 0.5], [0.3, 0.7]]},
            n_nodes=2,
        )
        s1 = SumNode(children=[layer], weights=[0.3, 0.7])

        categorical_nodes = [
            CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 2, "p": [0.5, 0.5]}),
            CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 2, "p": [0.3, 0.7]}),
        ]
        s2 = SumNode(children=categorical_nodes, weights=[0.3, 0.7])

        data = np.array([[0], [1], [0]])

        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))


    def test_layer_likelihood_2(self):

        layer = CondCategoricalLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"k": [2, 2], "p": [[0.5, 0.5], [0.3, 0.7]]},
            n_nodes=2
        )
        p1 = ProductNode(children=[layer])

        categorical_nodes = [
            CondCategorical(Scope([0], [2]), cond_f=lambda data: {"k": 2, "p": [0.5, 0.5]}),
            CondCategorical(Scope([1], [2]), cond_f=lambda data: {"k": 2, "p": [0.3, 0.7]}),
        ]
        p2 = ProductNode(children=categorical_nodes)

        data = np.array([[0, 1], [1, 1], [0, 0]])

        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
