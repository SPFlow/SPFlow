from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_binomial import (
    CondBinomial as BaseCondBinomial,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_binomial import (
    CondBinomial,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from typing import Callable

import torch
import numpy as np

import random
import unittest


class TestBinomial(unittest.TestCase):
    def test_initialization(self):

        binomial = CondBinomial(Scope([0], [1]), n=1)
        self.assertTrue(binomial.cond_f is None)
        binomial = CondBinomial(Scope([0], [1]), n=1, cond_f=lambda x: {"p": 0.5})
        self.assertTrue(isinstance(binomial.cond_f, Callable))

        # n = 0
        binomial = CondBinomial(Scope([0], [1]), 0, 0.5)
        # n < 0
        self.assertRaises(Exception, CondBinomial, Scope([0]), -1)
        # n float
        self.assertRaises(Exception, CondBinomial, Scope([0]), 0.5)
        # n = inf and n = nan
        self.assertRaises(Exception, CondBinomial, Scope([0]), np.inf)
        self.assertRaises(Exception, CondBinomial, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, CondBinomial, Scope([]), 1)
        self.assertRaises(Exception, CondBinomial, Scope([0, 1], [2]), 1)
        self.assertRaises(Exception, CondBinomial, Scope([0]), 1)

    def test_retrieve_params(self):

        # Valid parameters for Binomial distribution: p in [0,1], n in N U {0}

        binomial = CondBinomial(Scope([0], [1]), n=1)

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
        binomial.set_cond_f(
            lambda data: {
                "p": torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
            }
        )
        self.assertRaises(
            Exception,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        binomial.set_cond_f(
            lambda data: {
                "p": torch.nextafter(torch.tensor(0.0), -torch.tensor(1.0))
            }
        )
        self.assertRaises(
            Exception,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # p = inf and p = nan
        binomial.set_cond_f(lambda data: {"p": torch.tensor(float("inf"))})
        self.assertRaises(
            Exception,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        binomial.set_cond_f(lambda data: {"p": torch.tensor(float("nan"))})
        self.assertRaises(
            Exception,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(CondBinomial.accepts([([FeatureTypes.Discrete], Scope([0], [1]))]))

        # Bernoulli feature type class (should reject)
        self.assertFalse(CondBinomial.accepts([([FeatureTypes.Binomial], Scope([0], [1]))]))

        # Bernoulli feature type instance
        self.assertTrue(CondBinomial.accepts([([FeatureTypes.Binomial(n=3)], Scope([0], [1]))]))

        # invalid feature type
        self.assertFalse(CondBinomial.accepts([([FeatureTypes.Continuous], Scope([0], [1]))]))

        # non-conditional scope
        self.assertFalse(CondBinomial.accepts([([FeatureTypes.Binomial(n=3)], Scope([0]))]))

        # scope length does not match number of types
        self.assertFalse(CondBinomial.accepts([([FeatureTypes.Binomial(n=3)], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(CondBinomial.accepts([([FeatureTypes.Binomial(n=3), FeatureTypes.Binomial(n=3)], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        CondBinomial.from_signatures([([FeatureTypes.Binomial(n=3)], Scope([0], [1]))])
        CondBinomial.from_signatures([([FeatureTypes.Binomial(n=3, p=0.75)], Scope([0], [1]))])

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(ValueError, CondBinomial.from_signatures, [([FeatureTypes.Discrete], Scope([0], [1]))])

        # Bernoulli feature type class
        self.assertRaises(ValueError, CondBinomial.from_signatures, [([FeatureTypes.Binomial], Scope([0], [1]))])

        # invalid feature type
        self.assertRaises(ValueError, CondBinomial.from_signatures, [([FeatureTypes.Continuous], Scope([0], [1]))])

        # conditional scope
        self.assertRaises(ValueError, CondBinomial.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, CondBinomial.from_signatures, [([FeatureTypes.Discrete], Scope([0, 1], [2]))])

        # multivariate signature
        self.assertRaises(ValueError, CondBinomial.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1], [2]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondBinomial))

        # make sure leaf is correctly inferred
        self.assertEqual(CondBinomial, AutoLeaf.infer([([FeatureTypes.Binomial(n=3)], Scope([0], [1]))]))

        # make sure AutoLeaf can return correctly instantiated object
        binomial = AutoLeaf([([FeatureTypes.Binomial(n=3)], Scope([0], [1]))])
        self.assertTrue(isinstance(binomial, CondBinomial))
        self.assertEqual(binomial.n, 3)

    def test_structural_marginalization(self):

        binomial = CondBinomial(Scope([0], [1]), 1)

        self.assertTrue(marginalize(binomial, [1]) is not None)
        self.assertTrue(marginalize(binomial, [0]) is None)

    def test_base_backend_conversion(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = CondBinomial(Scope([0], [1]), n)
        node_binomial = BaseCondBinomial(Scope([0], [1]), n)

        # check conversion from torch to python
        self.assertTrue(
            np.all(
                torch_binomial.scopes_out == toBase(torch_binomial).scopes_out
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.all(
                node_binomial.scopes_out == toTorch(node_binomial).scopes_out
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
