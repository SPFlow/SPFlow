from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.binomial import Binomial
from spflow.base.inference.nodes.leaves.parametric.binomial import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestBinomial(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Binomial distribution: p in [0,1], n in N U {0}

        # p = 0
        binomial = Binomial(Scope([0]), 1, 0.0)
        # p = 1
        binomial = Binomial(Scope([0]), 1, 1.0)
        # p < 0 and p > 1
        self.assertRaises(
            Exception, Binomial, Scope([0]), 1, np.nextafter(1.0, 2.0)
        )
        self.assertRaises(
            Exception, Binomial, Scope([0]), 1, np.nextafter(0.0, -1.0)
        )
        # p = inf and p = nan
        self.assertRaises(Exception, Binomial, Scope([0]), 1, np.inf)
        self.assertRaises(Exception, Binomial, Scope([0]), 1, np.nan)

        # n = 0
        binomial = Binomial(Scope([0]), 0, 0.5)
        # n < 0
        self.assertRaises(Exception, Binomial, Scope([0]), -1, 0.5)
        # n float
        self.assertRaises(Exception, Binomial, Scope([0]), 0.5, 0.5)
        # n = inf and n = nan
        self.assertRaises(Exception, Binomial, Scope([0]), np.inf, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0]), np.nan, 0.5)

        # invalid scopes
        self.assertRaises(Exception, Binomial, Scope([]), 1, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0, 1]), 1, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0], [1]), 1, 0.5)

    def test_structural_marginalization(self):

        binomial = Binomial(Scope([0]), 1, 0.5)

        self.assertTrue(marginalize(binomial, [1]) is not None)
        self.assertTrue(marginalize(binomial, [0]) is None)


if __name__ == "__main__":
    unittest.main()
