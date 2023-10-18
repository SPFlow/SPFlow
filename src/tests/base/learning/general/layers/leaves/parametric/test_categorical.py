import random
import unittest

import numpy as np

from spflow.base.learning.general.layers.leaves.parametric.categorical import maximum_likelihood_estimation
from spflow.base.structure.spn import CategoricalLayer
from spflow.meta.data import Scope


class TestCategorical(unittest.TestCase):
    def test_mle(self):

        np.random.seed(0)
        random.seed(0)

        layer = CategoricalLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack([
            np.random.multinomial(n=1, pvals=[0.5, 0.5], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
            np.random.multinomial(n=1, pvals=[0.2, 0.8], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
        ])

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(np.allclose(layer.p, np.array([[0.5, 0.5], [0.2, 0.8]]), atol=1e-2, rtol=1e-3))


    def test_weighted_mle(self):

        layer = CategoricalLayer(scope=[Scope([0]), Scope([1])], n_nodes=3)

        
        data = np.hstack([
            np.vstack([
            np.random.multinomial(n=1, pvals=[0.5, 0.5], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
            np.random.multinomial(n=1, pvals=[0.3, 0.7], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
        ]), 
            np.vstack([
            np.random.multinomial(n=1, pvals=[0.2, 0.8], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1), 
            np.random.multinomial(n=1, pvals=[0.8, 0.2], size=(10000, 1)).reshape(-1, 2).argmax(axis=1).reshape(-1, 1)
        ])
        ])
        weights = np.concatenate([np.ones(10000), np.zeros(10000)]).reshape(-1)

        maximum_likelihood_estimation(layer, data, weights)

        self.assertTrue(np.allclose(layer.p, np.array([[0.5, 0.5], [0.2, 0.8]]), atol=0.01))


if __name__ == "__main__":
    unittest.main()




