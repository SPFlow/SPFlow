from spflow.base.structure.nodes.leaves.parametric import NegativeBinomial
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestNegativeBinomial(unittest.TestCase):
    def test_likelihood(self):

        # ----- configuration 1 -----
        n = 10
        p = 0.4

        negative_binomial = NegativeBinomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [5], [10]])
        targets = np.array([[0.000104858], [0.0163238], [0.0585708]])

        probs = likelihood(negative_binomial, data, SPN())
        log_probs = log_likelihood(negative_binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        n = 20
        p = 0.3

        negative_binomial = NegativeBinomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [10], [20]])
        targets = np.array([[0.0000000000348678], [0.0000197282], [0.00191757]])

        probs = likelihood(negative_binomial, data, SPN())
        log_probs = log_likelihood(negative_binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def initialization(self):

        # p = 1
        negative_binomial = NegativeBinomial([0], 1, 1.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data, SPN())
        log_probs = log_likelihood(negative_binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p = 0
        self.assertRaises(Exception, NegativeBinomial, [0], 1, 0.0)

        # 1 >= p > 0
        NegativeBinomial([0], 1, np.nextafter(0.0, 1.0))

        # p < 0 and p > 1
        self.assertRaises(Exception, NegativeBinomial, [0], 1, np.nextafter(1.0, 2.0))
        self.assertRaises(Exception, NegativeBinomial, [0], 1, np.nextafter(0.0, -1.0))

        # p inf, nan
        self.assertRaises(Exception, NegativeBinomial, [0], 1, np.inf)
        self.assertRaises(Exception, NegativeBinomial, [0], 1, np.nan)

        # n = 0
        NegativeBinomial([0], 0.0, 1.0)

        # n < 0
        self.assertRaises(Exception, NegativeBinomial, [0], np.nextafter(0.0, -1.0), 1.0)

        # n inf, nan
        self.assertRaises(Exception, NegativeBinomial, [0], np.inf, 1.0)
        self.assertRaises(Exception, NegativeBinomial, [0], np.nan, 1.0)

        # TODO: n float

        # set parameters to None manually
        negative_binomial.p = None
        self.assertRaises(Exception, likelihood, SPN(), negative_binomial, data)
        negative_binomial.n = None
        self.assertRaises(Exception, likelihood, SPN(), negative_binomial, data)

        # invalid scope lengths
        self.assertRaises(Exception, NegativeBinomial, [], 1, 0.5)
        self.assertRaises(Exception, NegativeBinomial, [0,1], 1, 0.5)

    def test_support(self):
    
        n = 20
        p = 0.3

        negative_binomial = NegativeBinomial([0], n, p)

        data = np.array([[np.nextafter(0.0, -1.0)], [0.0]])

        probs = likelihood(negative_binomial, data, SPN())
        log_probs = log_likelihood(negative_binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs[0] == 0.0))
        self.assertTrue(all(probs[1] != 0.0))


if __name__ == "__main__":
    unittest.main()
