from spflow.torch.utils.rankdata import rankdata
from scipy.stats import rankdata as scipy_rankdata

import torch
import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_rankdata_min(self):
        
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        # generate data
        data = np.random.randn(100, 5)
        # make sure that some elements are identical (to check tie-breaking)
        data[:10] = data[-10:]

        ranks_scipy = scipy_rankdata(data, axis=0, method='min')
        ranks_torch = rankdata(torch.tensor(data), method='min')

        self.assertTrue(np.all(ranks_scipy == ranks_torch.numpy()))
    
    def test_rankdata_max(self):
        
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        # generate data
        data = np.random.randn(100, 5)
        # make sure that some elements are identical (to check tie-breaking)
        data[:10] = data[-10:]

        ranks_scipy = scipy_rankdata(data, axis=0, method='max')
        ranks_torch = rankdata(torch.tensor(data), method='max')

        self.assertTrue(np.all(ranks_scipy == ranks_torch.numpy()))
    
    def test_rankdata_average(self):
        
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        # generate data
        data = np.random.randn(100, 5)
        # make sure that some elements are identical (to check tie-breaking)
        data[:10] = data[-10:]

        ranks_scipy = scipy_rankdata(data, axis=0, method='average')
        ranks_torch = rankdata(torch.tensor(data), method='average')

        self.assertTrue(np.all(ranks_scipy == ranks_torch.numpy()))


if __name__ == "__main__":
    unittest.main()