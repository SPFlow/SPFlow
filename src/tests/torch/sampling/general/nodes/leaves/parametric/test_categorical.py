import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.sampling import sample
from spflow.torch.structure.spn import Categorical


class TestCategorical(unittest.TestCase):

    def test_sampling(self):

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        categorical = Categorical(Scope([0]), k=2, p=[0.0, 1.0])

        data = torch.tensor([[torch.nan], [torch.nan], [torch.nan]])

        samples = sample(categorical, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~samples.isnan()] == 1))


    
    if __name__ == "__main__":
        torch.set_default_dtype(torch.float64)
        unittest.main()