import random
import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Categorical
from spflow.meta.data import Scope


class TestCategorical(unittest.TestCase):
    def test_likelihood_k2(self):

        k = 2
        p = [random.random() for i in range(k)]
        p = [x/sum(p) for x in p]

        categorical = Categorical(Scope([0]), k=k, p=p)

        # create test inputs/outputs
        data = np.array([[n] for n in range(k)])
        targets = np.array([[p[n]] for n in range(k)])

        probs = likelihood(categorical, data)
        log_probs = log_likelihood(categorical, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    
    def test_likelihood_k10(self):
        
        k = 10
        p = [random.random() for i in range(k)]
        p = [x/sum(p) for x in p]

        categorical = Categorical(Scope([0]), k=k, p=p)

        # create test inputs/outputs
        data = np.array([[n] for n in range(k)])
        targets = np.array([[p[n]] for n in range(k)])

        probs = likelihood(categorical, data)
        log_probs = log_likelihood(categorical, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))


    def test_likelihood_p_0(self):

        categorical = Categorical(Scope([0]), k=2, p=[1.0, 0.0])

        data = np.array([[0], [1]])
        targets = np.array([[1.0], [0.0]])
        
        probs = likelihood(categorical, data)
        log_probs = log_likelihood(categorical, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))


    def test_likelihood_p_none(self):

        categorical = Categorical(Scope([0]), k=1, p=[1.0])

        data = np.array([0])
        categorical.k = 0
        categorical.p = None
        self.assertRaises(Exception, likelihood, categorical, data)


    def test_likelihood_marginalization(self):

        k = 2
        p = [random.random() for i in range(k)]
        p = [x/sum(p) for x in p]

        categorical = Categorical(Scope([0]), k=k, p=p)

        data = np.array([[np.nan]])

        probs = likelihood(categorical, data)
        log_probs = log_likelihood(categorical, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))


    def test_support(self):

        # Support for Categorical distribution: integers {0, 1, ..., k-1}
        
        k = 2
        p = [random.random() for i in range(k)]
        p = [x/sum(p) for x in p]

        categorical = Categorical(Scope([0]), k=k, p=p)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, categorical, np.array([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, categorical, np.array([[np.inf]]))

        # check valid integers inside valid range
        data = np.array([[n] for n in range(k)])
        log_likelihood(categorical, data)

        # check valid integers outside valid ranges
        self.assertRaises(ValueError, log_likelihood, categorical, np.array([[-1]]))
        self.assertRaises(ValueError, log_likelihood, categorical, np.array([[k]]))

        # check invalid values
        self.assertRaises(ValueError, log_likelihood, categorical, np.array([[np.nextafter(0.0, -1.0)]]))
        self.assertRaises(ValueError, log_likelihood, categorical, np.array([[np.nextafter(0.0, 1.0)]]))
        self.assertRaises(ValueError, log_likelihood, categorical, np.array([[np.nextafter(1.0, 2.0)]]))
        self.assertRaises(ValueError, log_likelihood, categorical, np.array([[np.nextafter(1.0, 0.0)]]))


if __name__ == "__main__":
    unittest.main()
