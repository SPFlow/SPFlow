from spflow.meta.scope.scope import Scope
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.torch.sampling.nodes.leaves.parametric.hypergeometric import sample
from spflow.torch.sampling.module import sample

import torch
import unittest


class TestHypergeometric(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sampling_1(self):

       # ----- configuration 1 -----
        N = 500
        M = 100
        n = 50

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)

        self.assertTrue(torch.isclose(samples.mean(dim=0), torch.tensor(n*M)/N, atol=0.01, rtol=0.1))

    def test_sampling_2(self):

        # ----- configuration 2 -----
        N = 100
        M = 50
        n = 10

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)

        self.assertTrue(torch.isclose(samples.mean(dim=0), torch.tensor(n*M)/N, atol=0.01, rtol=0.1))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()