from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_binomial import (
    CondBinomial,
)
from typing import Callable

import numpy as np
import unittest


class TestCondBinomial(unittest.TestCase):
    def test_initialization(self):

        binomial = CondBinomial(Scope([0]), n=2)
        self.assertTrue(binomial.cond_f is None)
        binomial = CondBinomial(Scope([0]), n=2, cond_f=lambda x: {"p": 0.5})
        self.assertTrue(isinstance(binomial.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondBinomial, Scope([]), n=2)
        self.assertRaises(Exception, CondBinomial, Scope([0, 1]), n=2)
        self.assertRaises(Exception, CondBinomial, Scope([0], [1]), n=2)

        # Valid parameters for Binomial distribution: n in N U {0}

        # n = 0
        binomial = CondBinomial(Scope([0]), 0)
        # n < 0
        self.assertRaises(Exception, CondBinomial, Scope([0]), -1)
        # n float
        self.assertRaises(Exception, CondBinomial, Scope([0]), 0.5)
        # n = inf and n = nan
        self.assertRaises(Exception, CondBinomial, Scope([0]), np.inf)
        self.assertRaises(Exception, CondBinomial, Scope([0]), np.nan)

    def test_retrieve_params(self):

        # Valid parameters for Binomial distribution: p in [0,1]

        binomial = CondBinomial(Scope([0]), n=2)

        # p = 0
        binomial.set_cond_f(lambda data: {"p": 0.0})
        self.assertTrue(
            binomial.retrieve_params(np.array([[1.0]]), DispatchContext())
            == 0.0
        )
        # p = 1
        binomial.set_cond_f(lambda data: {"p": 1.0})
        self.assertTrue(
            binomial.retrieve_params(np.array([[1.0]]), DispatchContext())
            == 1.0
        )
        # p < 0 and p > 1
        binomial.set_cond_f(lambda data: {"p": np.nextafter(1.0, 2.0)})
        self.assertRaises(
            ValueError,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        binomial.set_cond_f(lambda data: {"p": np.nextafter(0.0, -1.0)})
        self.assertRaises(
            ValueError,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # p = inf and p = nan
        binomial.set_cond_f(lambda data: {"p": np.inf})
        self.assertRaises(
            ValueError,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        binomial.set_cond_f(lambda data: {"p": np.nan})
        self.assertRaises(
            ValueError,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_structural_marginalization(self):

        binomial = CondBinomial(Scope([0]), 1, lambda data: {"p": 0.5})

        self.assertTrue(marginalize(binomial, [1]) is not None)
        self.assertTrue(marginalize(binomial, [0]) is None)


if __name__ == "__main__":
    unittest.main()
