import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import log_likelihood
from spflow.torch.learning import em, expectation_maximization, maximum_likelihood_estimation
from spflow.torch.structure.spn import Categorical, ProductNode, SumNode


class TestCategorical(unittest.TestCase):

    def test_mle(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Categorical(Scope([0]), k=2)

        data = torch.tensor(np.random.multinomial(n=1, pvals=[0.3, 0.7], size=(10000,)).argmax(axis=1).reshape(-1, 1))

        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(torch.allclose(leaf.p, torch.tensor([0.3, 0.7]), atol=1e-2, rtol=1e-3))


    def test_mle_only_nans(self): 

        leaf = Categorical(Scope([0]))

        data = torch.tensor([[float("nan")], [float("nan")]])

        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy="ignore")


    def test_mle_invalid_support(self):

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Categorical(Scope([0]))

        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf")]]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[-0.1]]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[2]]))


    def test_mle_nan_strategy_none(self):

        categorical = Categorical(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, categorical, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=None)

    
    def test_mle_nan_strategy_ignore(self):

        categorical = Categorical(Scope([0]))
        maximum_likelihood_estimation(categorical, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy="ignore")
        self.assertTrue(torch.allclose(categorical.p, torch.tensor([1./3, 2./3])))


    def test_mle_nan_strategy_invalid(self):

        categorical = Categorical(Scope([0]))

        self.assertRaises(ValueError, maximum_likelihood_estimation, categorical, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy="invalid_string")
        self.assertRaises(ValueError, maximum_likelihood_estimation, categorical, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=1)


    def test_weighted_mle(self):

        categorical = Categorical(Scope([0]))

        data = torch.tensor(np.hstack([
            np.random.multinomial(n=1, pvals=[0.2, 0.8], size=(10000, 1)).reshape(10000, 2).argmax(axis=1), 
            np.random.multinomial(n=1, pvals=[0.5, 0.5], size=(10000, 1)).reshape(10000, 2).argmax(axis=1)
        ])).reshape(-1, 1)
        weights = torch.concat([torch.ones(10000), torch.zeros(10000)]).reshape(-1, 1)

        print(data.shape)
        print(weights.shape)

        maximum_likelihood_estimation(categorical, data, weights)

        self.assertTrue(torch.allclose(categorical.p, torch.tensor([0.2, 0.8]), atol=1e-3, rtol=1e-2))


    def test_em_step(self):    
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Categorical(Scope([0]))
        data = torch.tensor(np.random.multinomial(n=1, pvals=[0.3, 0.7], size=(10000,)).argmax(axis=1).reshape(-1, 1))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(leaf.p, torch.tensor([0.3, 0.7]), atol=1e-2, rtol=1e-3))
    

    def test_em_product_of_categoricals(self):
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Categorical(Scope([0]))
        l2 = Categorical(Scope([1]))
        prod_node = ProductNode([l1, l2])

        data = torch.tensor(np.hstack([
            np.random.multinomial(n=1, pvals=[0.3, 0.7], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
            np.random.multinomial(n=1, pvals=[0.2, 0.8], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
        ]))

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(torch.allclose(l1.p, torch.tensor([0.3, 0.7]), atol=1e-3, rtol=1e-2))
        self.assertTrue(torch.allclose(l2.p, torch.tensor([0.2, 0.8]), atol=1e-3, rtol=1e-2))

    

    def test_em_sum_of_categoricals(self):
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Categorical(Scope([0]), k=2, p=[0.6, 0.4])
        l2 = Categorical(Scope([0]), k=2, p=[0.4, 0.6])
        sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(np.vstack([
            np.random.multinomial(n=1, pvals=[0.2, 0.8], size=(10000,)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
            np.random.multinomial(n=1, pvals=[0.5, 0.5], size=(10000,)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
        ]))
        

        expectation_maximization(sum_node, data, max_steps=10)

        # optimal p
        p_opt = torch.tensor([torch.sum(data == k) / data.shape[0] for k in range(len(torch.unique(data)))])
        # total p represented by mixture
        p1 = sum_node.weights * l1.p 
        p2 = sum_node.weights * l2.p

        p_em = torch.tensor([p1[0]+p2[0], p1[1]+p2[1]])
        self.assertTrue(torch.allclose(p_opt, p_em, atol=5e-2, rtol=1e-1))

    

if __name__ == "__main__":
    unittest.main()



        



        