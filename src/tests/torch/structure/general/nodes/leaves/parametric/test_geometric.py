import random
import unittest

import numpy as np
import torch

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import Geometric as BaseGeometric
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure import AutoLeaf, marginalize, toBase, toTorch
from spflow.torch.structure.spn import Geometric


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
        self.assertTrue(
            Geometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # Geometric feature type class
        self.assertTrue(
            Geometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Geometric])]
            )
        )

        # Geometric feature type instance
        self.assertTrue(
            Geometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Geometric(0.5)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Geometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # conditional scope
        self.assertFalse(
            Geometric.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Geometric.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        geometric = Geometric.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
        )
        self.assertTrue(torch.isclose(geometric.p, torch.tensor(0.5)))

        geometric = Geometric.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Geometric])]
        )
        self.assertTrue(torch.isclose(geometric.p, torch.tensor(0.5)))

        geometric = Geometric.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)])]
        )
        self.assertTrue(torch.isclose(geometric.p, torch.tensor(0.75)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Geometric.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Geometric.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Geometric.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Geometric))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Geometric,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.Geometric])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        geometric = AutoLeaf(
            [FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)])]
        )
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
