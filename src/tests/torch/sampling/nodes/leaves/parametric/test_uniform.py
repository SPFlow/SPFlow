from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.torch.sampling.nodes.leaves.parametric.uniform import sample
from spflow.torch.sampling.module import sample

import torch
import numpy as np
import random
import unittest


class TestUniform(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- a = -1.0, b = 2.5 -----

        uniform = Uniform(Scope([0]), -1.0, 2.5)
        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(uniform, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))

        samples = sample(uniform, 1000)
        self.assertTrue(all((samples >= -1.0) & (samples <= 2.5)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()