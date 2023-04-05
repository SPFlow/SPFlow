import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves import CondBernoulli
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestCondBernoulli(unittest.TestCase):
    def test_sampling_1(self):

        # ----- p = 0 -----

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.0})

        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~tl_isnan(samples)] == 0.0))

    def test_sampling_2(self):

        # ----- p = 1 -----

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 1.0})

        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2], [[0], [0]]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~tl_isnan(samples)] == 1.0))

    def test_sampling_3(self):

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.5})

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            bernoulli,
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
