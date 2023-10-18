import unittest

import numpy as np

from spflow.base.sampling import sample
from spflow.base.structure.spn import CondCategorical
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestCondCategorical(unittest.TestCase):
    def test_sampling_1(self):

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 1, "p": [1.0]})

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(condCategorical, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 0))


    def test_sampling_2(self):

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 2, "p": [0.0, 1.0]})

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(condCategorical, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 1))


    def test_sampling_3(self):

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 1, "p": [1.0]})

        self.assertRaises(ValueError, sample, condCategorical, np.array([[0]]), sampling_ctx=SamplingContext([1]))


if __name__ == "__main__":
    unittest.main()