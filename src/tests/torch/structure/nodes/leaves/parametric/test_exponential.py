from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.exponential import (
    Exponential as BaseExponential,
)
from spflow.base.inference.nodes.leaves.parametric.exponential import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.exponential import (
    Exponential,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.exponential import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestExponential(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Exponential distribution: l>0

        # l > 0
        Exponential(
            Scope([0]), torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        )
        # l = 0 and l < 0
        self.assertRaises(Exception, Exponential, Scope([0]), 0.0)
        self.assertRaises(
            Exception,
            Exponential,
            Scope([0]),
            torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)),
        )
        # l = inf and l = nan
        self.assertRaises(Exception, Exponential, Scope([0]), np.inf)
        self.assertRaises(Exception, Exponential, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Exponential, Scope([]), 0.5)
        self.assertRaises(Exception, Exponential, Scope([0, 1]), 0.5)
        self.assertRaises(Exception, Exponential, Scope([0], [1]), 0.5)

    def test_structural_marginalization(self):

        exponential = Exponential(Scope([0]), 1.0)

        self.assertTrue(marginalize(exponential, [1]) is not None)
        self.assertTrue(marginalize(exponential, [0]) is None)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            Exponential.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # Exponential feature type class
        self.assertTrue(
            Exponential.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Exponential])]
            )
        )

        # Exponential feature type instance
        self.assertTrue(
            Exponential.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Exponential(1.0)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Exponential.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # conditional scope
        self.assertFalse(
            Exponential.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Exponential.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        exponential = Exponential.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
        )
        self.assertTrue(torch.isclose(exponential.l, torch.tensor(1.0)))

        exponential = Exponential.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Exponential])]
        )
        self.assertTrue(torch.isclose(exponential.l, torch.tensor(1.0)))

        exponential = Exponential.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Exponential(l=1.5)])]
        )
        self.assertTrue(torch.isclose(exponential.l, torch.tensor(1.5)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Exponential.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Exponential.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Exponential.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Exponential))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Exponential,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.Exponential])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        exponential = AutoLeaf(
            [FeatureContext(Scope([0]), [FeatureTypes.Exponential(l=1.5)])]
        )
        self.assertTrue(isinstance(exponential, Exponential))
        self.assertTrue(torch.isclose(exponential.l, torch.tensor(1.5)))

    def test_base_backend_conversion(self):

        l = random.random()

        torch_exponential = Exponential(Scope([0]), l)
        node_exponential = BaseExponential(Scope([0]), l)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_exponential.get_params()]),
                np.array([*toBase(torch_exponential).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_exponential.get_params()]),
                np.array([*toTorch(node_exponential).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
