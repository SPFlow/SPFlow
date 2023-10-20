import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_bernoulli import Bernoulli
from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_isnan

tc = unittest.TestCase()

def test_sampling_0(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- p = 0 -----

    bernoulli = Bernoulli(Scope([0]), 0.0)

    data = tl.tensor([[float("nan")], [float("nan")], [float("nan")]], dtype=tl.float64)

    samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2]))

    tc.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))
    tc.assertTrue(all(samples[~tl_isnan(samples)] == 0.0))

def test_sampling_1(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- p = 1 -----

    bernoulli = Bernoulli(Scope([0]), 1.0)

    data = tl.tensor([[float("nan")], [float("nan")], [float("nan")]], dtype=tl.float64)

    samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2], [[0], [0]]))

    tc.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))
    tc.assertTrue(all(samples[~tl_isnan(samples)] == 1.0))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    # ----- p = 0 -----

    bernoulli = Bernoulli(Scope([0]), 0.0)

    data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]], dtype=tl.float64)

    samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2]))
    notNans = samples[~tl_isnan(samples)]

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            bernoulli_updated = updateBackend(bernoulli)
            samples_updated = sample(bernoulli_updated, tl.tensor(data, dtype=tl.float64), sampling_ctx=SamplingContext([0, 2]))
            # check conversion from torch to python
            tc.assertTrue(all(tl_isnan(samples) == tl_isnan(samples_updated)))
            tc.assertTrue(all(tl_toNumpy(notNans) == tl_toNumpy(samples_updated[~tl_isnan(samples_updated)])))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
