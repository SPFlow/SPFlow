import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.modules.module import sample
from spflow.structure.general.node.leaf.general_log_normal import LogNormal
from spflow.torch.structure.general.node.leaf.log_normal import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_sampling_1(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- mean = 0.0, std = 1.0 -----

    log_normal = LogNormal(Scope([0]), 0.0, 1.0)

    data = tl.tensor([[float("nan")], [float("nan")], [float("nan")]], dtype=tl.float32)

    samples = sample(log_normal, data, sampling_ctx=SamplingContext([0, 2]))

    tc.assertTrue(all(tle.isnan(samples) == tl.tensor([[False], [True], [False]])))

    samples = sample(log_normal, 1000)
    tc.assertTrue(
        np.isclose(
            tl.mean(samples),
            np.exp(tl.tensor(0.0 + (1.0**2 / 2.0))),
            rtol=0.1,
        )
    )


def test_sampling_2(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- mean = 1.0, std = 0.5 -----

    log_normal = LogNormal(Scope([0]), 1.0, 0.5)

    samples = sample(log_normal, 1000)
    tc.assertTrue(
        np.isclose(
            tl.mean(samples),
            np.exp(tl.tensor(1.0 + (0.5**2 / 2.0))),
            rtol=0.1,
        )
    )


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # ----- mean = 0.0, std = 1.0 -----

    log_normal = LogNormal(Scope([0]), 0.0, 1.0)

    data = tl.tensor([[float("nan")], [float("nan")], [float("nan")]], dtype=tl.float32)

    samples = sample(log_normal, data, sampling_ctx=SamplingContext([0, 2]))
    notNans = samples[~tle.isnan(samples)]

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            log_normal_updated = updateBackend(log_normal)
            samples_updated = sample(
                log_normal_updated, tl.tensor(data, dtype=tl.float32), sampling_ctx=SamplingContext([0, 2])
            )
            # check conversion from torch to python
            tc.assertTrue(all(tle.isnan(samples) == tle.isnan(samples_updated)))
            tc.assertTrue(
                all(tle.toNumpy(notNans) == tle.toNumpy(samples_updated[~tle.isnan(samples_updated)]))
            )


def test_change_dtype(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

    layer = LogNormal(Scope([0]), 0.0, 1.0)
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

    layer = LogNormal(Scope([0]), 0.0, 1.0)
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
