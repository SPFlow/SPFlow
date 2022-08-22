from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.base.sampling.nodes.leaves.parametric.geometric import sample
from spflow.base.sampling.module import sample

import numpy as np

import unittest


class TestGeometric(unittest.TestCase):
    def test_sampling_1(self):

        # ----- p = 1.0 -----

        geometric = Geometric(Scope([0]), 1.0)

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(geometric, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 1.0))
    
    def test_sampling_2(self):

        # ----- p = 0.5 -----

        geometric = Geometric(Scope([0]), 0.5)

        samples = sample(geometric, 1000)

        self.assertTrue(np.isclose(samples.mean(), np.array(1.0/ 0.5), rtol=0.1))

    def test_sampling_3(self):

        # ----- p = 0.8 -----

        geometric = Geometric(Scope([0]), 0.8)

        samples = sample(geometric, 1000)

        self.assertTrue(np.isclose(samples.mean(), np.array(1.0/0.8), rtol=0.1))

    def test_sampling_4(self):

        geometric = Geometric(Scope([0]), 0.8)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(ValueError, sample, geometric, np.array([[0]]), sampling_ctx=SamplingContext([1]))


if __name__ == "__main__":
    unittest.main()