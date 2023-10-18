import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.learning import em, expectation_maximization, maximum_likelihood_estimation
from spflow.torch.structure.spn import CategoricalLayer, ProductNode, SumNode
from spflow.torch.inference import log_likelihood


class TestCategorical(unittest.TestCase):

    def test_mle(self):

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = CategoricalLayer(scope=[Scope([0]), Scope([1])])

        data = torch.tensor(np.hstack(
            [
                np.random.multinomial(n=1, pvals=[0.5, 0.5], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
                np.random.multinomial(n=1, pvals=[0.3, 0.7], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
            ]
        ))

        maximum_likelihood_estimation(layer, data)

        self.assertTrue(torch.allclose(layer.p, torch.tensor([[0.5, 0.5], [0.3, 0.7]]), atol=1e-2, rtol=1e-3))


    def test_mle_only_nans(self):

        layer = CategoricalLayer(scope=[Scope([0]), Scope([1])], n_nodes=1)

        data = torch.tensor([[float("nan"), float("nan"), float("nan")]])

        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, data, nan_strategy="ignore")


    def test_mle_invalid_support(self):

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = CategoricalLayer(Scope([0]), k=2, p=[0.3, 0.7])

        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("inf")]]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[-0.1]]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[2]]))


    def test_mle_nan_strategy_none(self):

        layer = CategoricalLayer(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=None)


    def test_mle_nan_strategy_callable(self):

        layer = CategoricalLayer(Scope([0]))
        maximum_likelihood_estimation(layer, torch.tensor([[1], [0], [1]]), nan_strategy=lambda x: x)


    def test_mle_nan_strategy_invalid(self):

        layer = CategoricalLayer(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy="invalid")
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=1)


    def test_weighted_mle(self):

        layer = CategoricalLayer([Scope([0]), Scope([1])], k=2, n_nodes=3)

        data = torch.tensor(
            np.hstack(
                [
                    np.vstack([
                        np.random.multinomial(n=1, pvals=[0.3, 0.7], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
                        np.random.multinomial(n=1, pvals=[0.1, 0.9], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
                    ]), 
                    np.vstack([
                        np.random.multinomial(n=1, pvals=[0.6, 0.4], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
                        np.random.multinomial(n=1, pvals=[0.8, 0.2], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
                    ])
                ]
            )
        )

        weights = torch.concat([torch.ones(10000), torch.zeros(10000)]).reshape(-1)

        maximum_likelihood_estimation(layer, data, weights)

        self.assertTrue(torch.allclose(layer.p, torch.tensor([[0.3, 0.7], [0.6, 0.4]]), atol=1e-3, rtol=1e-2))


    def test_em_step(self):
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = CategoricalLayer([Scope([0]), Scope([1])])
        data = torch.tensor(np.hstack(
            [
                np.random.multinomial(n=1, pvals=[0.5, 0.5], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
                np.random.multinomial(n=1, pvals=[0.3, 0.7], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
            ]
        ))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(layer, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(layer.p, torch.tensor([[0.5, 0.5], [0.3, 0.7]]), atol=1e-2, rtol=1e-3))

    
    
    def test_em_product_of_categoricals(self):
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = CategoricalLayer([Scope([0]), Scope([1])])
        prod_node = ProductNode([layer])

        data = torch.tensor(np.hstack(
            [
                np.random.multinomial(n=1, pvals=[0.5, 0.5], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
                np.random.multinomial(n=1, pvals=[0.3, 0.7], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
            ]
        ))

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(torch.allclose(layer.p, torch.tensor([[0.5, 0.5], [0.3, 0.7]]), atol=1e-2, rtol=1e-3))
    

    def test_em_sum_of_categoricals(self):
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = CategoricalLayer(Scope([0]), n_nodes=2, p=[[0.4, 0.6], [0.6, 0.4]])
        sum_node = SumNode([leaf], weights=[0.5, 0.5])

        data = torch.tensor(np.vstack([
            np.random.multinomial(n=1, pvals=[0.2, 0.8], size=(10000,)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
            np.random.multinomial(n=1, pvals=[0.5, 0.5], size=(10000,)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
        ]))

        expectation_maximization(sum_node, data, max_steps=10)

        # optimal p
        p_opt = torch.tensor([torch.sum(data == k) / data.shape[0] for k in range(len(torch.unique(data)))])
        # total p represented by mixture
        p1 = sum_node.weights * leaf.p[0] 
        p2 = sum_node.weights * leaf.p[1]

        p_em = torch.tensor([p1[0]+p2[0], p1[1]+p2[1]])

        self.assertTrue(torch.allclose(p_opt, p_em, atol=5e-2, rtol=1e-1))
    

if __name__ == "__main__":
    unittest.main()
