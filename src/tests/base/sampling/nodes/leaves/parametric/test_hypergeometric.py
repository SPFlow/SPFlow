from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.base.sampling.nodes.leaves.parametric.hypergeometric import sample
from spflow.base.sampling.module import sample

import numpy as np

import unittest


class TestHypergeometric(unittest.TestCase):
    def test_sampling(self):

        hypergeometric = Hypergeometric(Scope([0]), 10, 10, 10)

        self.assertRaises(NotImplementedError, sample, hypergeometric)


if __name__ == "__main__":
    unittest.main()