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
    
        start_end = random.random()

        self.assertRaises(Exception, Uniform, [0], start_end, start_end)
        self.assertRaises(Exception, Uniform, [0], start_end, np.nextafter(start_end, -1.0))
        self.assertRaises(Exception, Uniform, [0], np.inf, 0.0)
        self.assertRaises(Exception, Uniform, [0], np.nan, 0.0)
        self.assertRaises(Exception, Uniform, [0], 0.0, np.inf)
        self.assertRaises(Exception, Uniform, [0], 0.0, np.nan)

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


if __name__ == "__main__":
    unittest.main()

