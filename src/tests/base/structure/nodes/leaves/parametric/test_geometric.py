from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.base.inference.nodes.leaves.parametric.geometric import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestGeometric(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Geometric distribution: p in (0,1]

        # p = 0
        self.assertRaises(Exception, Geometric, Scope([0]), 0.0)
        # p = inf and p = nan
        self.assertRaises(Exception, Geometric, Scope([0]), np.inf)
        self.assertRaises(Exception, Geometric, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Geometric, Scope([]), 0.5)
        self.assertRaises(Exception, Geometric, Scope([0, 1]), 0.5)
        self.assertRaises(Exception, Geometric, Scope([0], [1]), 0.5)

    def test_structural_marginalization(self):

        geometric = Geometric(Scope([0]), 0.5)

        self.assertTrue(marginalize(geometric, [1]) is not None)
        self.assertTrue(marginalize(geometric, [0]) is None)


if __name__ == "__main__":
    unittest.main()
