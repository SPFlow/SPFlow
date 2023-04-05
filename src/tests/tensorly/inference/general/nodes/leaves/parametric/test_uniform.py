import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import Uniform
from spflow.meta.data import Scope


class TestUniform(unittest.TestCase):
    def test_likelihood(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        uniform = Uniform(Scope([0]), start, end)

        # create test inputs/outputs
        data = tl.tensor(
            [
                [np.nextafter(start, -tl.inf)],
                [start],
                [(start + end) / 2.0],
                [end],
                [np.nextafter(end, tl.inf)],
            ]
        )
        targets = tl.tensor(
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

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

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
        data = tl.tensor([[tl.nan, tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(uniform, data)
        log_probs = log_likelihood(uniform, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Uniform distribution: floats [a,b] or (-inf,inf)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        # ----- with support outside the interval -----
        uniform = Uniform(Scope([0]), 1.0, 2.0, support_outside=True)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, uniform, tl.tensor([[-tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, uniform, tl.tensor([[tl.inf]]))

        # check valid floats in [start, end]
        log_likelihood(uniform, tl.tensor([[1.0]]))
        log_likelihood(uniform, tl.tensor([[1.5]]))
        log_likelihood(uniform, tl.tensor([[2.0]]))

        # check valid floats outside [start, end]
        log_likelihood(uniform, tl.tensor([[np.nextafter(1.0, -1.0)]]))
        log_likelihood(uniform, tl.tensor([[np.nextafter(2.0, 3.0)]]))

        # ----- without support outside the interval -----
        uniform = Uniform(Scope([0]), 1.0, 2.0, support_outside=False)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, uniform, tl.tensor([[-tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, uniform, tl.tensor([[tl.inf]]))

        # check valid floats in [start, end]
        log_likelihood(uniform, tl.tensor([[1.0]]))
        log_likelihood(uniform, tl.tensor([[1.5]]))
        log_likelihood(uniform, tl.tensor([[2.0]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            uniform,
            tl.tensor([[np.nextafter(1.0, -1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            uniform,
            tl.tensor([[np.nextafter(2.0, 3.0)]]),
        )


if __name__ == "__main__":
    unittest.main()
