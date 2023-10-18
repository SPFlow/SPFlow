import random
import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import CondCategorical
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestCondCategorical(unittest.TestCase):
    def test_likelihood_no_p(self):

        condCategorical = CondCategorical(Scope([0], [1]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[0]]))

    def test_likelihood_module_cond_f(self):

        k = 2
        p = [random.random() for i in range(k)]
        p = [x/sum(p) for x in p]
        cond_f = lambda data: {"k": k, "p": p}

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[n] for n in range(k)])
        targets = np.array([[p[n]] for n in range(k)])

        probs = likelihood(condCategorical, data)
        log_probs = log_likelihood(condCategorical, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))


    def test_likelihood_args_p(self):

        condCategorical = CondCategorical(Scope([0], [1]))

        k = 2
        p = [random.random() for i in range(k)]
        p = [x/sum(p) for x in p]
        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[condCategorical] = {"k": k, "p": p}

        # create test inputs/outputs
        data = np.array([[n] for n in range(k)])
        targets = np.array([[p[n]] for n in range(k)])

        probs = likelihood(condCategorical, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(condCategorical, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs))) 
        self.assertTrue(np.allclose(probs, targets))


    def test_likelihood_cond_f(self):

        condCategorical = CondCategorical(Scope([0], [1]))

        k = 2
        p = [random.random() for i in range(k)]
        p = [x/sum(p) for x in p]
        cond_f = lambda data: {"k": k, "p": p}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[condCategorical] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[n] for n in range(k)])
        targets = np.array([[p[n]] for n in range(k)])

        probs = likelihood(condCategorical, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(condCategorical, data, dispatch_ctx=dispatch_ctx)


    def test_likelihood_p_0(self):

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 2, "p": [1.0, 0.0]})

        data = np.array([[0], [1]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(condCategorical, data)
        log_probs = log_likelihood(condCategorical, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    
    def test_likelihood_p_none(self):

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 1, "p": None})
        
        data = np.array([[0]])
        
        self.assertRaises(Exception, likelihood, condCategorical, data)


    def test_likelihood_marginalization(self):

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 2, "p": [0.5, 0.5]})
        data = np.array([[np.nan]])

        probs = likelihood(condCategorical, data)
        log_probs = log_likelihood(condCategorical, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))


    def test_support(self):

        # Support for Categorical distribution: {0, 1, ..., k-1}

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 2, "p": [0.5, 0.5]})

        # check inf
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[np.inf]]))

        # check valid integer in valid range
        log_likelihood(condCategorical, np.array([[0], [1]]))

        # check valid integer outside valid range
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[-1]]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[2]]))

        # check invalid values
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[np.nextafter(0.0, -1.0)]]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[np.nextafter(0.0, 1.0)]]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[np.nextafter(1.0, 2.0)]]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[np.nextafter(1.0, 0.0)]]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, np.array([[0.5]]))


if __name__ == "__main__":
    unittest.main()



