from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.cond_poisson import CondPoissonLayer
from spflow.base.inference.layers.leaves.parametric.cond_poisson import log_likelihood
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import CondPoisson
from spflow.base.inference.nodes.leaves.parametric.cond_poisson import log_likelihood
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood, likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_likelihood_no_l(self):

        poisson = CondPoissonLayer(Scope([0]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, poisson, np.array([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {'l': [1, 1]}

        poisson = CondPoissonLayer(Scope([0]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))
    
    def test_likelihood_args_l(self):

        poisson = CondPoissonLayer(Scope([0]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[poisson] = {'l': [1, 1]}

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        poisson = CondPoissonLayer(Scope([0]), n_nodes=2)

        cond_f = lambda data: {'l': np.array([1, 1])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[poisson] = {'cond_f': cond_f}

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_layer_likelihood_1(self):

        poisson_layer = CondPoissonLayer(scope=Scope([0]), cond_f=lambda data: {'l': [0.8, 0.3]}, n_nodes=2)
        s1 = SPNSumNode(children=[poisson_layer], weights=[0.3, 0.7])

        poisson_nodes = [CondPoisson(Scope([0]), cond_f=lambda data: {'l': 0.8}), CondPoisson(Scope([0]), cond_f=lambda data: {'l': 0.3})]
        s2 = SPNSumNode(children=poisson_nodes, weights=[0.3, 0.7])

        data = np.array([[1], [5], [3]])

        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))
    
    def test_layer_likelihood_2(self):

        poisson_layer = CondPoissonLayer(scope=[Scope([0]), Scope([1])], cond_f=lambda data: {'l': [0.8, 0.3]})
        p1 = SPNProductNode(children=[poisson_layer])

        poisson_nodes = [CondPoisson(Scope([0]), cond_f=lambda data: {'l': 0.8}), CondPoisson(Scope([1]), cond_f=lambda data: {'l': 0.3})]
        p2 = SPNProductNode(children=poisson_nodes)

        data = np.array([[1, 6], [5, 3], [3, 7]])

        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()