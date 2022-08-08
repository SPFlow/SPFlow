from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.base.sampling.nodes.leaves.parametric.multivariate_gaussian import sample
from spflow.base.sampling.module import sample

import numpy as np

import unittest


class TestMultivariateGaussian(unittest.TestCase):
    def test_sampling(self):

        multivariate_gaussian = MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))

        self.assertRaises(NotImplementedError, sample, multivariate_gaussian)


if __name__ == "__main__":
    unittest.main()