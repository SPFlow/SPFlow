from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.torch.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.torch.sampling.nodes.leaves.parametric.gamma import sample
from spflow.torch.sampling.module import sample

import torch
import unittest


class TestGamma(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sampling_1(self):

        # ----- alpha = 1, beta = 1 -----

        gamma = Gamma(Scope([0]), 1.0, 1.0)

        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(gamma, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))

        samples = sample(gamma, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(1.0 / 1.0), rtol=0.1))

    def test_sampling_2(self):

        # ----- alpha = 0.5, beta = 1.5 -----

        gamma = Gamma(Scope([0]), 0.5, 1.5)

        samples = sample(gamma, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(0.5 / 1.5), rtol=0.1))

    def test_sampling_3(self):

        # ----- alpha = 1.5, beta = 0.5 -----

        gamma = Gamma(Scope([0]), 1.5, 0.5)

        samples = sample(gamma, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(1.5 / 0.5), rtol=0.1))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()