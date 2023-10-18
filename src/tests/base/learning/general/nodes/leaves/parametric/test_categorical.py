import random
import unittest

import numpy as np

from spflow.base.learning import maximum_likelihood_estimation
from spflow.base.structure.spn import Categorical
from spflow.meta.data import Scope


class TestCategorical(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Categorical(Scope([0]), k=2)

        # simulate data
        probs = np.array([0.3, 0.7])
        data = np.random.multinomial(n=1, pvals=probs, size=(10000, 1)).reshape(10000, 2).argmax(axis=1).reshape(-1, 1)

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(np.allclose(leaf.p, probs, atol=1e-2, rtol=1e-3))

    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Categorical(Scope([0]), k=4)

        # simulate data
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        data = np.random.multinomial(n=1, pvals=probs, size=(10000, 1)).reshape(10000, 4).argmax(axis=1).reshape(-1, 1)

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(np.allclose(leaf.p, probs, atol=1e-2, rtol=1e-3))


    # def test_mle_edge_1(self):
    # does not make sense for categorical distributions

    #     # set seed
    #     np.random.seed(0)
    #     random.seed(0)

    #     leaf = Categorical(Scope([0]), k=1)

    #     # simulate data
    #     data = np.random.multinomial(n=1, pvals=[1.0], size=(100, 1)).reshape(100, 1).argmax(axis=1).reshape(-1, 1)

    #     # perform MLE
    #     maximum_likelihood_estimation(leaf, data)

    #     self.assertTrue(np.all(leaf.p < 1.0))

    
    def test_mle_only_nans(self):
        
        leaf = Categorical(Scope([0]))

        # simulate data
        data = np.array([[np.nan], [np.nan]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy="ignore")


    def test_mle_invalid_support(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Categorical(Scope([0]), k=2)

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.inf]]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[-1]]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[2]]))


    def test_mle_nan_strategy_none(self):

        leaf = Categorical(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy=None)


    def test_mle_nan_strategy_ignore(self):

        leaf = Categorical(Scope([0]))
        maximum_likelihood_estimation(leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy="ignore")
        self.assertTrue(np.allclose(leaf.p, [1.0/3.0, 2.0/3.0]))


    def test_mle_nan_strategy_invalid(self):

        leaf = Categorical(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy="invalid_string")
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy=1)


    def test_weighted_mle(self):

        leaf = Categorical(Scope([0]))

        data = np.hstack([
            np.random.multinomial(n=1, pvals=[0.2, 0.8], size=(10000, 1)).reshape(10000, 2).argmax(axis=1), 
            np.random.multinomial(n=1, pvals=[0.8, 0.2], size=(10000, 1)).reshape(10000, 2).argmax(axis=1)
        ]).reshape(-1, 1)
        weights = np.concatenate([np.ones(10000), np.zeros(10000)]).reshape(-1, 1)

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(np.allclose(leaf.p, np.array([0.2, 0.8]), atol=0.01))
                        

if __name__ == "__main__":
    unittest.main()

        

