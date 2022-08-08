from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
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

    def test_sampling(self):

        hypergeometric = Hypergeometric(Scope([0]), 10, 10, 10)

        self.assertRaises(NotImplementedError, sample, hypergeometric)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()