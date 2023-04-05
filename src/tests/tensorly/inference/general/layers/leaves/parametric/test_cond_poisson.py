import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import (
    ProductNode,
    SumNode,
)
from spflow.tensorly.structure.general.nodes.leaves import CondPoisson
from spflow.tensorly.structure.general.layers.leaves import CondPoissonLayer
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestNode(unittest.TestCase):
    def test_likelihood_no_l(self):

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"l": [1, 1]}

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_l(self):

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[poisson] = {"l": [1, 1]}

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        poisson = CondPoissonLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"l": tl.tensor([1, 1])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[poisson] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.367879, 0.367879], [0.18394, 0.18394], [0.00306566, 0.00306566]])

        probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_layer_likelihood_1(self):

        poisson_layer = CondPoissonLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"l": [0.8, 0.3]},
            n_nodes=2,
        )
        s1 = SumNode(children=[poisson_layer], weights=[0.3, 0.7])

        poisson_nodes = [
            CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 0.8}),
            CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 0.3}),
        ]
        s2 = SumNode(children=poisson_nodes, weights=[0.3, 0.7])

        data = tl.tensor([[1], [5], [3]])

        self.assertTrue(tl.all(log_likelihood(s1, data) == log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        poisson_layer = CondPoissonLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"l": [0.8, 0.3]},
        )
        p1 = ProductNode(children=[poisson_layer])

        poisson_nodes = [
            CondPoisson(Scope([0], [2]), cond_f=lambda data: {"l": 0.8}),
            CondPoisson(Scope([1], [2]), cond_f=lambda data: {"l": 0.3}),
        ]
        p2 = ProductNode(children=poisson_nodes)

        data = tl.tensor([[1, 6], [5, 3], [3, 7]])

        self.assertTrue(tl.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
