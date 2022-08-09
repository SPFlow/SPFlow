from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.base.sampling.nodes.leaves.parametric.hypergeometric import sample
from spflow.base.sampling.module import sample

import numpy as np

import unittest


class TestHypergeometric(unittest.TestCase):
    def test_sampling_1(self):

       # ----- configuration 1 -----
        N = 500
        M = 100
        n = 50

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)

        self.assertTrue(np.isclose(samples.mean(axis=0), np.array(n*M)/N, atol=0.01, rtol=0.1))

    def test_sampling_2(self):

        # ----- configuration 2 -----
        N = 100
        M = 50
        n = 10

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)

        self.assertTrue(np.isclose(samples.mean(axis=0), np.array(n*M)/N, atol=0.01, rtol=0.1))

if __name__ == "__main__":
    unittest.main()