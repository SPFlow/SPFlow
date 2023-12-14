import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.node.leaf.general_cond_negative_binomial import CondNegativeBinomial
from spflow.torch.structure.general.node.leaf.negative_binomial import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_isnan

tc = unittest.TestCase()

def test_sampling_1(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- n = 1, p = 1.0 -----

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 1, cond_f=lambda data: {"p": 1.0})
    data = tl.tensor([[float("nan")], [float("nan")], [float("nan")]], dtype=tl.float32)

    samples = sample(negative_binomial, data, sampling_ctx=SamplingContext([0, 2]))

    tc.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))
    tc.assertTrue(all(samples[~tl_isnan(samples)] == 0.0))

def test_sampling_2(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- n = 10, p = 0.3 -----

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 10, cond_f=lambda data: {"p": 0.3})

    samples = sample(negative_binomial, 1000)
    tc.assertTrue(np.isclose(tl.mean(samples), tl.tensor(10 * (1 - 0.3) / 0.3), rtol=0.1))

def test_sampling_3(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- n = 5, p = 0.8 -----

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 5, cond_f=lambda data: {"p": 0.8})

    samples = sample(negative_binomial, 1000)
    tc.assertTrue(np.isclose(tl.mean(samples), tl.tensor(5 * (1 - 0.8) / 0.8), rtol=0.1))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- n = 1, p = 1.0 -----

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 1, cond_f=lambda data: {"p": 1.0})
    data = tl.tensor([[float("nan")], [float("nan")], [float("nan")]], dtype=tl.float32)

    samples = sample(negative_binomial, data, sampling_ctx=SamplingContext([0, 2]))
    notNans = samples[~tl_isnan(samples)]

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            negative_binomial_updated = updateBackend(negative_binomial)
            samples_updated = sample(negative_binomial_updated, tl.tensor(data, dtype=tl.float32), sampling_ctx=SamplingContext([0, 2]))
            # check conversion from torch to python
            tc.assertTrue(all(tl_isnan(samples) == tl_isnan(samples_updated)))
            tc.assertTrue(all(tl_toNumpy(notNans) == tl_toNumpy(samples_updated[~tl_isnan(samples_updated)])))

def test_change_dtype(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

    layer = CondNegativeBinomial(Scope([0], [1]), 1, cond_f=lambda data: {"p": 1.0})
    # get samples
    data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]], dtype=tl.float32)

    samples = sample(layer, data, sampling_ctx=SamplingContext([0, 2]))

    tc.assertTrue(samples.dtype == tl.float32)
    layer.to_dtype(tl.float64)
    data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]], dtype=tl.float64)

    samples = sample(layer, data, sampling_ctx=SamplingContext([0, 2]))
    tc.assertTrue(samples.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = CondNegativeBinomial(Scope([0], [1]), 1, cond_f=lambda data: {"p": 1.0})
    # get samples
    data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]], dtype=tl.float32)

    samples = sample(layer, data, sampling_ctx=SamplingContext([0, 2]))

    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    tc.assertTrue(samples.device.type == "cpu")
    layer.to_device(cuda)
    data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]], dtype=tl.float32, device=cuda)

    samples = sample(layer, data, sampling_ctx=SamplingContext([0, 2]))
    tc.assertTrue(samples.device.type == "cuda")



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
