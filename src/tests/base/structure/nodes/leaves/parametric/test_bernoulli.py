from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.bernoulli import Bernoulli
from spflow.base.inference.nodes.leaves.parametric.bernoulli import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood
from typing import Callable

import numpy as np
import unittest
import random


class TestBernoulli(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        # p = 0
        bernoulli = Bernoulli(Scope([0]), 0.0)
        # p = 1
        bernoulli = Bernoulli(Scope([0]), 1.0)
        # p < 0 and p > 1
        self.assertRaises(
            Exception, Bernoulli, Scope([0]), np.nextafter(1.0, 2.0)
        )
        self.assertRaises(
            Exception, Bernoulli, Scope([0]), np.nextafter(0.0, -1.0)
        )
        # p = inf and p = nan
        self.assertRaises(Exception, Bernoulli, Scope([0]), np.inf)
        self.assertRaises(Exception, Bernoulli, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Bernoulli, Scope([]), 0.5)
        self.assertRaises(Exception, Bernoulli, Scope([0, 1]), 0.5)
        self.assertRaises(Exception, Bernoulli, Scope([0], [1]), 0.5)

    def test_structural_marginalization(self):

        bernoulli = Bernoulli(Scope([0]), 0.5)

        self.assertTrue(marginalize(bernoulli, [1]) is not None)
        self.assertTrue(marginalize(bernoulli, [0]) is None)


if __name__ == "__main__":
    unittest.main()
