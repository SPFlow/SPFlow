import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.sampling import sample
from spflow.tensorly.sampling import sample
#from spflow.torch.structure.spn import Hypergeometric
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_hypergeometric import Hypergeometric
from spflow.torch.structure.general.nodes.leaves.parametric.hypergeometric import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_isnan


class TestHypergeometric(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sampling_1(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- configuration 1 -----
        N = 500
        M = 100
        n = 50

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)

        self.assertTrue(
            torch.isclose(
                samples.mean(dim=0),
                torch.tensor(n * M) / N,
                atol=0.01,
                rtol=0.1,
            )
        )

    def test_sampling_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- configuration 2 -----
        N = 100
        M = 50
        n = 10

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)

        self.assertTrue(
            torch.isclose(
                samples.mean(dim=0),
                torch.tensor(n * M) / N,
                atol=0.01,
                rtol=0.1,
            )
        )

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- configuration 1 -----
        N = 500
        M = 100
        n = 50

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)
        #notNans = samples[~tl_isnan(samples)]

        # make sure that probabilities match python backend probabilities
        for backend in backends:
            tl.set_backend(backend)
            hypergeometric_updated = updateBackend(hypergeometric)
            samples_updated = sample(hypergeometric_updated, 1000)
            self.assertTrue(np.isclose(tl_toNumpy(samples).mean(axis=0), tl_toNumpy(samples_updated).mean(axis=0), atol=0.01, rtol=0.1))
            # check conversion from torch to python
            #self.assertTrue(all(tl_isnan(samples) == tl_isnan(samples_updated)))
            #self.assertTrue(all(tl_toNumpy(notNans) == tl_toNumpy(samples_updated[~tl_isnan(samples_updated)])))



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
