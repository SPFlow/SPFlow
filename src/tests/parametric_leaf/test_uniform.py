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
        # start = inf and start = nan
        self.assertRaises(Exception, Uniform, [0], np.inf, 0.0)
        self.assertRaises(Exception, Uniform, [0], np.nan, 0.0)
        # end = inf and end = nan
        self.assertRaises(Exception, Uniform, [0], 0.0, np.inf)
        self.assertRaises(Exception, Uniform, [0], 0.0, np.nan)

        # dummy distribution and data
        uniform = Uniform([0], 0.0, 1.0)
        data = np.random.rand(1,3)

        # set parameters to None manually
        uniform.end = None
        self.assertRaises(Exception, likelihood, SPN(), uniform, data)
        uniform.start = None
        self.assertRaises(Exception, likelihood, SPN(), uniform, data)

        # invalid scope length
        self.assertRaises(Exception, Uniform, [], 0.0, 1.0)
        self.assertRaises(Exception, Uniform, [0,1], 0.0, 1.0)

    def test_support(self):

        # Support for Uniform distribution: [a,b] (TODO: R?)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None
        #
        #   outside support -> 0 (or NaN?)

        l = random.random()

        uniform = Uniform([0], 1.0, 2.0)

        # edge cases (-inf,inf), values outside of [1,2]
        data = np.array([[-np.inf], [np.nextafter(1.0, -np.inf)], [np.nextafter(2.0, np.inf)], [np.inf]])
        targets = np.zeros((4,1))

        probs = likelihood(uniform, data, SPN())
        log_probs = log_likelihood(uniform, data, SPN())

        self.assertTrue(np.allclose(probs, targets))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))


if __name__ == "__main__":
    unittest.main()

