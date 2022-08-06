from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.log_normal import LogNormal
from spflow.base.inference.nodes.leaves.parametric.log_normal import log_likelihood
from spflow.base.inference.module import likelihood

import numpy as np
import unittest
import random


class TestLogNormal(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Log-Normal distribution: mean in (-inf,inf), stdev in (0,inf)

        # mean = +-inf and mean = nan
        self.assertRaises(Exception, LogNormal, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0]), -np.inf, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0]), np.nan, 1.0)

        mean = random.random()

        # stdev <= 0
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, 0.0)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.nextafter(0.0, -1.0))
        # stdev = +-inf and stdev = nan
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.inf)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, -np.inf)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.nan)

        # invalid scopes
        self.assertRaises(Exception, LogNormal, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0],[1]), 0.0, 1.0)

    def test_structural_marginalization(self):

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(log_normal, [1]) is not None)
        self.assertTrue(marginalize(log_normal, [0]) is None)

if __name__ == "__main__":
    unittest.main()
