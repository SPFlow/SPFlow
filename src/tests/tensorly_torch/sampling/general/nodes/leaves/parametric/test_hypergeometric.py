import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_hypergeometric import Hypergeometric
from spflow.torch.structure.general.nodes.leaves.parametric.hypergeometric import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_isnan

tc = unittest.TestCase()

def test_sampling_1(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    # ----- configuration 1 -----
    N = 500
    M = 100
    n = 50

    hypergeometric = Hypergeometric(Scope([0]), N, M, n)

    samples = sample(hypergeometric, 100000)

    tc.assertTrue(
        np.isclose(
            tl.mean(samples, axis=0),
            tl.tensor(n * M) / N,
            atol=0.01,
            rtol=0.1,
        )
    )

def test_sampling_2(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    # ----- configuration 2 -----
    N = 100
    M = 50
    n = 10

    hypergeometric = Hypergeometric(Scope([0]), N, M, n)

    samples = sample(hypergeometric, 100000)

    tc.assertTrue(
        np.isclose(
            tl.mean(samples, axis=0),
            tl.tensor(n * M) / N,
            atol=0.01,
            rtol=0.1,
        )
    )

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    # ----- configuration 1 -----
    N = 500
    M = 100
    n = 50

    hypergeometric = Hypergeometric(Scope([0]), N, M, n)

    samples = sample(hypergeometric, 100000)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            hypergeometric_updated = updateBackend(hypergeometric)
            samples_updated = sample(hypergeometric_updated, 1000)
            tc.assertTrue(np.isclose(tl_toNumpy(samples).mean(axis=0), tl_toNumpy(samples_updated).mean(axis=0), atol=0.01, rtol=0.1))
            # check conversion from torch to python



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
