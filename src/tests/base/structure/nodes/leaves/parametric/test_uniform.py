from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.base.inference.nodes.leaves.parametric.uniform import log_likelihood
from spflow.base.inference.module import likelihood

import numpy as np
import unittest
import random


class TestUniform(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Uniform distribution: a<b

        # start = end
        start_end = random.random()
        self.assertRaises(Exception, Uniform, Scope([0]), start_end, start_end)
        # start > end
        self.assertRaises(
            Exception,
            Uniform,
            Scope([0]),
            start_end,
            np.nextafter(start_end, -1.0),
        )
        # start = +-inf and start = nan
        self.assertRaises(Exception, Uniform, Scope([0]), np.inf, 0.0)
        self.assertRaises(Exception, Uniform, Scope([0]), -np.inf, 0.0)
        self.assertRaises(Exception, Uniform, Scope([0]), np.nan, 0.0)

        # end = +-inf and end = nan
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, np.inf)
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, -np.inf)
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Uniform, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, Uniform, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, Uniform, Scope([0], [1]))

    def test_structural_marginalization(self):

        uniform = Uniform(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(uniform, [1]) is not None)
        self.assertTrue(marginalize(uniform, [0]) is None)


if __name__ == "__main__":
    unittest.main()
