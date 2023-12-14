import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.node.leaf.general_gaussian import Gaussian
from spflow.torch.structure.general.node.leaf.gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_isnan

tc = unittest.TestCase()

def test_sampling(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    gaussian = Gaussian(Scope([0]), 0.0, 0.0005)

    data = tl.tensor([[float("nan")], [float("nan")], [float("nan")]], dtype=tl.float32)

    samples = sample(gaussian, data, sampling_ctx=SamplingContext([0, 2]))

    tc.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))

    # ----- verify samples -----
    samples = sample(gaussian, 1000)
    tc.assertTrue(np.isclose(tl.mean(samples), torch.Tensor([0.0]), atol=0.01))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    gaussian = Gaussian(Scope([0]), 0.0, 0.0005)

    data = tl.tensor([[float("nan")], [float("nan")], [float("nan")]], dtype=tl.float32)

    samples = sample(gaussian, data, sampling_ctx=SamplingContext([0, 2]))
    notNans = samples[~tl_isnan(samples)]

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            gaussian_updated = updateBackend(gaussian)
            samples_updated = sample(gaussian_updated, tl.tensor(data, dtype=tl.float32), sampling_ctx=SamplingContext([0, 2]))
            # check conversion from torch to python
            tc.assertTrue(all(tl_isnan(samples) == tl_isnan(samples_updated)))
            tc.assertTrue(all(tl_toNumpy(notNans) == tl_toNumpy(samples_updated[~tl_isnan(samples_updated)])))

def test_change_dtype(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

    layer = Gaussian(Scope([0]), 0.0, 0.0005)
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

    layer = Gaussian(Scope([0]), 0.0, 0.0005)
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
