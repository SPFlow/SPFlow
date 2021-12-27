from spflow.base.structure.nodes.leaves.parametric import Uniform
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest
import random


class TestUniform(unittest.TestCase):
    def test_likelihood(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        uniform = Uniform([0], start, end)

        # create test inputs/outputs
        data = np.array(
            [
                [np.nextafter(start, -np.inf)],
                [start],
                [(start + end) / 2.0],
                [end],
                [np.nextafter(end, np.inf)],
            ]
        )
        targets = np.array(
            [[0.0], [1.0 / (end - start)], [1.0 / (end - start)], [1.0 / (end - start)], [0.0]]
        )

        probs = likelihood(uniform, data, SPN())
        log_probs = log_likelihood(uniform, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        # Valid parameters for Uniform distribution: a<b

        # start = end
        start_end = random.random()
        self.assertRaises(Exception, Uniform, [0], start_end, start_end)
        # start > end
        self.assertRaises(Exception, Uniform, [0], start_end, np.nextafter(start_end, -1.0))
        # start = +-inf and start = nan
        self.assertRaises(Exception, Uniform, [0], np.inf, 0.0)
        self.assertRaises(Exception, Uniform, [0], -np.inf, 0.0)
        self.assertRaises(Exception, Uniform, [0], np.nan, 0.0)
        # end = +-inf and end = nan
        self.assertRaises(Exception, Uniform, [0], 0.0, np.inf)
        self.assertRaises(Exception, Uniform, [0], 0.0, -np.inf)
        self.assertRaises(Exception, Uniform, [0], 0.0, np.nan)

        # dummy distribution and data
        uniform = Uniform([0], 0.0, 1.0)
        data = np.random.rand(1, 3)

        # set parameters to None manually
        uniform.end = None
        self.assertRaises(Exception, likelihood, uniform, data, SPN())
        uniform.start = None
        self.assertRaises(Exception, likelihood, uniform, data, SPN())

        # invalid scope length
        self.assertRaises(Exception, Uniform, [], 0.0, 1.0)
        self.assertRaises(Exception, Uniform, [0, 1], 0.0, 1.0)

    def test_support(self):

        # Support for Uniform distribution: floats [a,b] or (-inf,inf)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        # ----- with support outside the interval -----
        uniform = Uniform([0], 1.0, 2.0, support_outside=True)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, uniform, np.array([[-np.inf]]), SPN())
        self.assertRaises(ValueError, log_likelihood, uniform, np.array([[np.inf]]), SPN())

        # check valid floats in [start, end]
        log_likelihood(uniform, np.array([[1.0]]), SPN())
        log_likelihood(uniform, np.array([[1.5]]), SPN())
        log_likelihood(uniform, np.array([[2.0]]), SPN())

        # check valid floats outside [start, end]
        log_likelihood(uniform, np.array([[np.nextafter(1.0, -1.0)]]), SPN())
        log_likelihood(uniform, np.array([[np.nextafter(2.0, 3.0)]]), SPN())

        # ----- without support outside the interval -----
        uniform = Uniform([0], 1.0, 2.0, support_outside=False)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, uniform, np.array([[-np.inf]]), SPN())
        self.assertRaises(ValueError, log_likelihood, uniform, np.array([[np.inf]]), SPN())

        # check valid floats in [start, end]
        log_likelihood(uniform, np.array([[1.0]]), SPN())
        log_likelihood(uniform, np.array([[1.5]]), SPN())
        log_likelihood(uniform, np.array([[2.0]]), SPN())

        # check invalid float values
        self.assertRaises(
            ValueError, log_likelihood, uniform, np.array([[np.nextafter(1.0, -1.0)]]), SPN()
        )
        self.assertRaises(
            ValueError, log_likelihood, uniform, np.array([[np.nextafter(2.0, 3.0)]]), SPN()
        )

    def test_marginalization(self):

        uniform = Uniform([0], 1.0, 2.0)
        data = np.array([[np.nan, np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(uniform, data, SPN())
        log_probs = log_likelihood(uniform, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))


if __name__ == "__main__":
    unittest.main()
