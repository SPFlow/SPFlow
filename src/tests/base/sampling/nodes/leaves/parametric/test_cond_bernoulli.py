from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.cond_bernoulli import CondBernoulli
from spflow.base.sampling.nodes.leaves.parametric.cond_bernoulli import sample

import numpy as np

import unittest


class TestCondBernoulli(unittest.TestCase):
    def test_sampling_1(self):

        # ----- p = 0 -----

        bernoulli = CondBernoulli(Scope([0]), cond_f=lambda data: {'p': 0.0})

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 0.0))

    def test_sampling_2(self):

        # ----- p = 1 -----

        bernoulli = CondBernoulli(Scope([0]), cond_f=lambda data: {'p': 1.0})

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2], [[0], [0]]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 1.0))

    def test_sampling_3(self):

        bernoulli = CondBernoulli(Scope([0]), cond_f=lambda data: {'p': 0.5})

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(ValueError, sample, bernoulli, np.array([[0]]), sampling_ctx=SamplingContext([1]))


if __name__ == "__main__":
    unittest.main()