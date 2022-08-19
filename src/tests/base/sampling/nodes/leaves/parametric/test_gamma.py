from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.base.sampling.nodes.leaves.parametric.gamma import sample
from spflow.base.sampling.module import sample

import numpy as np

import unittest


class TestGamma(unittest.TestCase):
    def test_sampling_1(self):

        # ----- alpha = 1, beta = 1 -----

        gamma = Gamma(Scope([0]), 1.0, 1.0)

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(gamma, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))

        samples = sample(gamma, 1000)
        self.assertTrue(np.isclose(samples.mean(), np.array(1.0 / 1.0), rtol=0.1))

    def test_sampling_2(self):

        # ----- alpha = 0.5, beta = 1.5 -----

        gamma = Gamma(Scope([0]), 0.5, 1.5)

        samples = sample(gamma, 1000)
        self.assertTrue(np.isclose(samples.mean(), np.array(0.5 / 1.5), rtol=0.1))

    def test_sampling_3(self):

        # ----- alpha = 1.5, beta = 0.5 -----

        gamma = Gamma(Scope([0]), 1.5, 0.5)

        samples = sample(gamma, 1000)
        self.assertTrue(np.isclose(samples.mean(), np.array(1.5 / 0.5), rtol=0.1))

    def test_sampling_4(self):

        gamma = Gamma(Scope([0]), 1.5, 0.5)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(ValueError, sample, gamma, np.array([[0]]), sampling_ctx=SamplingContext([1]))
        

if __name__ == "__main__":
    unittest.main()