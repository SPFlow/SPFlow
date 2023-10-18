import unittest

import numpy as np

from spflow.base.sampling import sample
from spflow.base.structure.spn import Categorical
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext

class TestCategorical(unittest.TestCase):
    def test_sampling_1(self):
        
        categorical = Categorical(Scope([0]), k=1, p=[1.0])

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(categorical, data, sampling_ctx = SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 0))


    def test_sampling_2(self):

        categorical = Categorical(Scope([0]), k=2, p=[0.0, 1.0])

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(categorical, data, sampling_ctx = SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 1))


    def test_sampling_3(self):

        categorical = Categorical(Scope([0]), k=3, p=[0.0, 0.0, 1.0])

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(categorical, data, sampling_ctx = SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 2))


    def test_sampling_4(self):

        categorical = Categorical(Scope([0]))

        self.assertRaises(ValueError, sample, categorical, np.array([[0]]), sampling_ctx=SamplingContext([1]))


if __name__ == "__main__":
    unittest.main()