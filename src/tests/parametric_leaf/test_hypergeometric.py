from spflow.base.structure.nodes.leaves.parametric import Hypergeometric
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestHypergeometric(unittest.TestCase):
    def test_initialization(self):

        # ----- configuration 1 -----
        N = 500
        M = 100
        n = 50

        hypergeometric = Hypergeometric([0], N, M, n)

        # create test inputs/outputs
        data = np.array([[5], [10], [15]])
        targets = np.array([[0.0257071], [0.147368], [0.0270206]])

        probs = likelihood(hypergeometric, data, SPN())
        log_probs = log_likelihood(hypergeometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        N = 100
        M = 50
        n = 10

        hypergeometric = Hypergeometric([0], N, M, n)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.00723683], [0.259334], [0.00059342]])

        probs = likelihood(hypergeometric, data, SPN())
        log_probs = log_likelihood(hypergeometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):
    
        self.assertRaises(Exception, Hypergeometric, -1, 1, 1)
        self.assertRaises(Exception, Hypergeometric, 1, -1, 1)
        self.assertRaises(Exception, Hypergeometric, 1, 2, 1)
        self.assertRaises(Exception, Hypergeometric, 1, 1, -1)
        self.assertRaises(Exception, Hypergeometric, 1, 1, 2)
        self.assertRaises(Exception, Hypergeometric, [0], np.inf, 1, 1)
        self.assertRaises(Exception, Hypergeometric, [0], np.nan, 1, 1)
        self.assertRaises(Exception, Hypergeometric, [0], 1, np.inf, 1)
        self.assertRaises(Exception, Hypergeometric, [0], 1, np.nan, 1)
        self.assertRaises(Exception, Hypergeometric, [0], 1, 1, np.inf)
        self.assertRaises(Exception, Hypergeometric, [0], 1, 1, np.nan)

        hypergeometric = Hypergeometric([0], N=500, M=100, n=50)
        data = np.array([[5], [10], [15]])

        # set parameters to None manually
        hypergeometric.n = None
        self.assertRaises(Exception, likelihood, SPN(), hypergeometric, data)
        hypergeometric.M = None
        self.assertRaises(Exception, likelihood, SPN(), hypergeometric, data)
        hypergeometric.N = None
        self.assertRaises(Exception, likelihood, SPN(), hypergeometric, data)

        # invalid scope lengths
        self.assertRaises(Exception, Hypergeometric, [], 1, 1, 1)
        self.assertRaises(Exception, Hypergeometric, [0,1], 1, 1, 1)

    def test_support(self):

        N = 15
        M = 10
        n = 10

        hypergeometric = Hypergeometric([0], N, M, n)

        # create test inputs/outputs
        data = np.array([[4], [11], [5], [10]])

        probs = likelihood(hypergeometric, data, SPN())
        log_probs = log_likelihood(hypergeometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs[:2] == 0))
        self.assertTrue(all(probs[2:] != 0))


if __name__ == "__main__":
    unittest.main()
