import random
import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Uniform
from spflow.meta.data import Scope


class TestUniform(unittest.TestCase):
    def test_likelihood(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        uniform = Uniform(Scope([0]), start, end)

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
            [
                [0.0],
                [1.0 / (end - start)],
                [1.0 / (end - start)],
                [1.0 / (end - start)],
                [0.0],
            ]
        )

        probs = likelihood(uniform, data)
        log_probs = log_likelihood(uniform, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_start_none(self):

        # dummy distribution and data
        uniform = Uniform(Scope([0]), 0.0, 1.0)
        data = np.random.rand(1, 3)

        # set parameter to None manually
        uniform.start = None
        self.assertRaises(Exception, likelihood, uniform, data)

    def test_likelihood_end_none(self):

        # dummy distribution and data
        uniform = Uniform(Scope([0]), 0.0, 1.0)
        data = np.random.rand(1, 3)

        # set parameter to None manually
        uniform.end = None
        self.assertRaises(Exception, likelihood, uniform, data)

    def test_likelihood_marginalization(self):

        uniform = Uniform(Scope([0]), 1.0, 2.0)
        data = np.array([[np.nan, np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(uniform, data)
        log_probs = log_likelihood(uniform, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Uniform distribution: floats [a,b] or (-inf,inf)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        # ----- with support outside the interval -----
        uniform = Uniform(Scope([0]), 1.0, 2.0, support_outside=True)

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, uniform, np.array([[-np.inf]])
        )
        self.assertRaises(
            ValueError, log_likelihood, uniform, np.array([[np.inf]])
        )

        # check valid floats in [start, end]
        log_likelihood(uniform, np.array([[1.0]]))
        log_likelihood(uniform, np.array([[1.5]]))
        log_likelihood(uniform, np.array([[2.0]]))

        # check valid floats outside [start, end]
        log_likelihood(uniform, np.array([[np.nextafter(1.0, -1.0)]]))
        log_likelihood(uniform, np.array([[np.nextafter(2.0, 3.0)]]))

        # ----- without support outside the interval -----
        uniform = Uniform(Scope([0]), 1.0, 2.0, support_outside=False)

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, uniform, np.array([[-np.inf]])
        )
        self.assertRaises(
            ValueError, log_likelihood, uniform, np.array([[np.inf]])
        )

        # check valid floats in [start, end]
        log_likelihood(uniform, np.array([[1.0]]))
        log_likelihood(uniform, np.array([[1.5]]))
        log_likelihood(uniform, np.array([[2.0]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            uniform,
            np.array([[np.nextafter(1.0, -1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            uniform,
            np.array([[np.nextafter(2.0, 3.0)]]),
        )


if __name__ == "__main__":
    unittest.main()
