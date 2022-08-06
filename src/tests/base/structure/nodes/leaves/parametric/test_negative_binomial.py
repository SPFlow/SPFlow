from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.negative_binomial import NegativeBinomial
from spflow.base.inference.nodes.leaves.parametric.negative_binomial import log_likelihood
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestNegativeBinomial(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Negative Binomial distribution: p in [0,1], n > 0

        # p < 0 and p > 1
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.nextafter(1.0, 2.0))
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.nextafter(0.0, -1.0))
        # p = +-inf and p = nan
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.inf)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, -np.inf)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.nan)

        # n = 0
        NegativeBinomial(Scope([0]), 0.0, 1.0)
        # n < 0
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), np.nextafter(0.0, -1.0), 1.0)
        # n = inf and = nan
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), np.nan, 1.0)
        
        # invalid scopes
        self.assertRaises(Exception, NegativeBinomial, Scope([]), 1, 0.5)
        self.assertRaises(Exception, NegativeBinomial, Scope([0, 1]), 1, 0.5)
        self.assertRaises(Exception, NegativeBinomial, Scope([0],[1]), 1, 0.5)

    def test_structural_marginalization(self):
        
        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)

        self.assertTrue(marginalize(negative_binomial, [1]) is not None)
        self.assertTrue(marginalize(negative_binomial, [0]) is None)

if __name__ == "__main__":
    unittest.main()
