from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.bernoulli import (
    Bernoulli as BaseBernoulli,
)
from spflow.base.inference.nodes.leaves.parametric.bernoulli import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.bernoulli import (
    Bernoulli,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.bernoulli import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestBernoulli(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        # p = 0
        bernoulli = Bernoulli(Scope([0]), 0.0)
        # p = 1
        bernoulli = Bernoulli(Scope([0]), 1.0)
        # p < 0 and p > 1
        self.assertRaises(
            Exception,
            Bernoulli,
            Scope([0]),
            torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)),
        )
        self.assertRaises(
            Exception,
            Bernoulli,
            Scope([0]),
            torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)),
        )
        # p = inf and p = nan
        self.assertRaises(Exception, Bernoulli, Scope([0]), np.inf)
        self.assertRaises(Exception, Bernoulli, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Bernoulli, Scope([]), 0.5)
        self.assertRaises(Exception, Bernoulli, Scope([0, 1]), 0.5)
        self.assertRaises(Exception, Bernoulli, Scope([0], [1]), 0.5)

    def test_accept(self):

        # discrete meta type
        self.assertTrue(Bernoulli.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # Bernoulli feature type class
        self.assertTrue(Bernoulli.accepts([([FeatureTypes.Bernoulli], Scope([0]))]))

        # Bernoulli feature type instance
        self.assertTrue(Bernoulli.accepts([([FeatureTypes.Bernoulli(0.5)], Scope([0]))]))

        # invalid feature type
        self.assertFalse(Bernoulli.accepts([([FeatureTypes.Continuous], Scope([0]))]))

        # conditional scope
        self.assertFalse(Bernoulli.accepts([([FeatureTypes.Discrete], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(Bernoulli.accepts([([FeatureTypes.Discrete], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(Bernoulli.accepts([([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        bernoulli = Bernoulli.from_signatures([([FeatureTypes.Discrete], Scope([0]))])
        self.assertTrue(torch.isclose(bernoulli.p, torch.tensor(0.5)))

        bernoulli = Bernoulli.from_signatures([([FeatureTypes.Bernoulli], Scope([0]))])
        self.assertTrue(torch.isclose(bernoulli.p, torch.tensor(0.5)))
    
        bernoulli = Bernoulli.from_signatures([([FeatureTypes.Bernoulli(p=0.75)], Scope([0]))])
        self.assertTrue(torch.isclose(bernoulli.p, torch.tensor(0.75)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, Bernoulli.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, Bernoulli.from_signatures, [([FeatureTypes.Discrete], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, Bernoulli.from_signatures, [([FeatureTypes.Discrete], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, Bernoulli.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Bernoulli))

        # make sure leaf is correctly inferred
        self.assertEqual(Bernoulli, AutoLeaf.infer([([FeatureTypes.Bernoulli], Scope([0]))]))

        # make sure AutoLeaf can return correctly instantiated object
        bernoulli = AutoLeaf([([FeatureTypes.Bernoulli(p=0.75)], Scope([0]))])
        self.assertTrue(isinstance(bernoulli, Bernoulli))
        self.assertTrue(torch.isclose(bernoulli.p, torch.tensor(0.75)))

    def test_structural_marginalization(self):

        bernoulli = Bernoulli(Scope([0]), 0.5)

        self.assertTrue(marginalize(bernoulli, [1]) is not None)
        self.assertTrue(marginalize(bernoulli, [0]) is None)

    def test_base_backend_conversion(self):

        p = random.random()

        torch_bernoulli = Bernoulli(Scope([0]), p)
        node_bernoulli = BaseBernoulli(Scope([0]), p)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_bernoulli.get_params()]),
                np.array([*toBase(torch_bernoulli).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_bernoulli.get_params()]),
                np.array([*toTorch(node_bernoulli).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
