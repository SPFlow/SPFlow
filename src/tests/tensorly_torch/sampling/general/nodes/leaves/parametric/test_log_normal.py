import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.sampling import sample
from spflow.tensorly.sampling import sample
#from spflow.torch.structure.spn import LogNormal
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_log_normal import LogNormal
from spflow.torch.structure.general.nodes.leaves.parametric.log_normal import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_isnan


class TestLogNormal(unittest.TestCase):
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

        # ----- mean = 0.0, std = 1.0 -----

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(log_normal, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))

        samples = sample(log_normal, 1000)
        self.assertTrue(
            torch.isclose(
                samples.mean(),
                torch.exp(torch.tensor(0.0 + (1.0**2 / 2.0))),
                rtol=0.1,
            )
        )

    def test_sampling_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- mean = 1.0, std = 0.5 -----

        log_normal = LogNormal(Scope([0]), 1.0, 0.5)

        samples = sample(log_normal, 1000)
        self.assertTrue(
            torch.isclose(
                samples.mean(),
                torch.exp(torch.tensor(1.0 + (0.5**2 / 2.0))),
                rtol=0.1,
            )
        )

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- mean = 0.0, std = 1.0 -----

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(log_normal, data, sampling_ctx=SamplingContext([0, 2]))
        notNans = samples[~tl_isnan(samples)]

        # make sure that probabilities match python backend probabilities
        for backend in backends:
            tl.set_backend(backend)
            log_normal_updated = updateBackend(log_normal)
            samples_updated = sample(log_normal_updated, tl.tensor(data, dtype=tl.float64), sampling_ctx=SamplingContext([0, 2]))
            # check conversion from torch to python
            self.assertTrue(all(tl_isnan(samples) == tl_isnan(samples_updated)))
            self.assertTrue(all(tl_toNumpy(notNans) == tl_toNumpy(samples_updated[~tl_isnan(samples_updated)])))



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
