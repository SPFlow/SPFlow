from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.geometric import (
    Geometric as BaseGeometric,
)
from spflow.base.inference.nodes.leaves.parametric.geometric import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.geometric import (
    Geometric,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.geometric import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestGeometric(unittest.TestCase):
    def test_initialiation(self):

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

    def test_accept(self):

        # discrete meta type
        self.assertTrue(Geometric.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # Geometric feature type class
        self.assertTrue(Geometric.accepts([([FeatureTypes.Geometric], Scope([0]))]))

        # Geometric feature type instance
        self.assertTrue(Geometric.accepts([([FeatureTypes.Geometric(0.5)], Scope([0]))]))

        # invalid feature type
        self.assertFalse(Geometric.accepts([([FeatureTypes.Continuous], Scope([0]))]))

        # conditional scope
        self.assertFalse(Geometric.accepts([([FeatureTypes.Discrete], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(Geometric.accepts([([FeatureTypes.Discrete], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(Geometric.accepts([([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        geometric = Geometric.from_signatures([([FeatureTypes.Discrete], Scope([0]))])
        self.assertTrue(torch.isclose(geometric.p, torch.tensor(0.5)))

        geometric = Geometric.from_signatures([([FeatureTypes.Geometric], Scope([0]))])
        self.assertTrue(torch.isclose(geometric.p, torch.tensor(0.5)))
    
        geometric = Geometric.from_signatures([([FeatureTypes.Geometric(p=0.75)], Scope([0]))])
        self.assertTrue(torch.isclose(geometric.p, torch.tensor(0.75)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, Geometric.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, Geometric.from_signatures, [([FeatureTypes.Discrete], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, Geometric.from_signatures, [([FeatureTypes.Discrete], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, Geometric.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Geometric))

        # make sure leaf is correctly inferred
        self.assertEqual(Geometric, AutoLeaf.infer([([FeatureTypes.Geometric], Scope([0]))]))

        # make sure AutoLeaf can return correctly instantiated object
        geometric = AutoLeaf([([FeatureTypes.Geometric(p=0.75)], Scope([0]))])
        self.assertTrue(isinstance(geometric, Geometric))
        self.assertTrue(torch.isclose(geometric.p, torch.tensor(0.75)))

    def test_base_backend_conversion(self):

        p = random.random()

        torch_geometric = Geometric(Scope([0]), p)
        node_geometric = BaseGeometric(Scope([0]), p)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_geometric.get_params()]),
                np.array([*toBase(torch_geometric).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_geometric.get_params()]),
                np.array([*toTorch(node_geometric).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
