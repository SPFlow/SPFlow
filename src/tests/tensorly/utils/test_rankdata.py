import random
import unittest

import numpy as np
import torch
from scipy.stats import rankdata as scipy_rankdata

from spflow.torch.utils import rankdata

tc = unittest.TestCase()

def test_rankdata_min(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # generate data
    data = np.random.randn(100, 5)
    # make sure that some elements are identical (to check tie-breaking)
    data[:10] = data[-10:]

    ranks_scipy = scipy_rankdata(data, axis=0, method="min")
    ranks_torch = rankdata(torch.tensor(data), method="min")

    tc.assertTrue(np.all(ranks_scipy == ranks_torch.numpy()))

def test_rankdata_max(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # generate data
    data = np.random.randn(100, 5)
    # make sure that some elements are identical (to check tie-breaking)
    data[:10] = data[-10:]

    ranks_scipy = scipy_rankdata(data, axis=0, method="max")
    ranks_torch = rankdata(torch.tensor(data), method="max")

    tc.assertTrue(np.all(ranks_scipy == ranks_torch.numpy()))

def test_rankdata_average(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # generate data
    data = np.random.randn(100, 5)
    # make sure that some elements are identical (to check tie-breaking)
    data[:10] = data[-10:]

    ranks_scipy = scipy_rankdata(data, axis=0, method="average")
    ranks_torch = rankdata(torch.tensor(data), method="average")

    tc.assertTrue(np.all(ranks_scipy == ranks_torch.numpy()))


if __name__ == "__main__":
    unittest.main()
