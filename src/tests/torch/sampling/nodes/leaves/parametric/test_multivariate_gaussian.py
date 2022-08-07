from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.torch.sampling.nodes.leaves.parametric.multivariate_gaussian import sample
from spflow.torch.sampling.module import sample

import torch
import numpy as np

import random
import unittest


class TestMultivariateGaussian(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sampling(self):

        multivariate_gaussian = MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))

        self.assertRaises(NotImplementedError, sample, multivariate_gaussian)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()