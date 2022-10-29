from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.exponential import (
    Exponential,
)
from spflow.base.inference.nodes.leaves.parametric.exponential import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestExponential(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Exponential distribution: l>0

        # l > 0
        exponential = Exponential(Scope([0]), np.nextafter(0.0, 1.0))
        # l = 0 and l < 0
        self.assertRaises(Exception, Exponential, Scope([0]), 0.0)
        self.assertRaises(
            Exception, Exponential, Scope([0]), np.nextafter(0.0, -1.0)
        )
        # l = inf and l = nan
        self.assertRaises(Exception, Exponential, Scope([0]), np.inf)
        self.assertRaises(Exception, Exponential, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Exponential, Scope([]), 1.0)
        self.assertRaises(Exception, Exponential, Scope([0, 1]), 1.0)
        self.assertRaises(Exception, Exponential, Scope([0], [1]), 1.0)

    def test_structural_marginalization(self):

        exponential = Exponential(Scope([0]), 1.0)

        self.assertTrue(marginalize(exponential, [1]) is not None)
        self.assertTrue(marginalize(exponential, [0]) is None)


if __name__ == "__main__":
    unittest.main()
