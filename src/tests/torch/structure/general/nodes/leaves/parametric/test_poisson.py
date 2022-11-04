from spflow.meta.data import Scope, FeatureTypes, FeatureContext
from spflow.torch.structure import AutoLeaf
from spflow.torch.structure.spn import Poisson
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.inference import log_likelihood
from spflow.base.structure.spn import Poisson as BasePoisson
from spflow.base.inference import log_likelihood

import torch
import numpy as np

import random
import unittest


class TestPoisson(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in pytorch)

        # l = 0
        Poisson(Scope([0]), 0.0)
        # l > 0
        Poisson(
            Scope([0]), torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        )
        # l = -inf and l = inf
        self.assertRaises(Exception, Poisson, Scope([0]), -np.inf)
        self.assertRaises(Exception, Poisson, Scope([0]), np.inf)
        # l = nan
        self.assertRaises(Exception, Poisson, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Poisson, Scope([]), 1)
        self.assertRaises(Exception, Poisson, Scope([0, 1]), 1)
        self.assertRaises(Exception, Poisson, Scope([0], [1]), 1)

    def test_structural_marginalization(self):

        poisson = Poisson(Scope([0]), 1.0)

        self.assertTrue(marginalize(poisson, [1]) is not None)
        self.assertTrue(marginalize(poisson, [0]) is None)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            Poisson.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # Poisson feature type class
        self.assertTrue(
            Poisson.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Poisson])]
            )
        )

        # Poisson feature type instance
        self.assertTrue(
            Poisson.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Poisson(1.0)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Poisson.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # conditional scope
        self.assertFalse(
            Poisson.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Poisson.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        poisson = Poisson.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
        )
        self.assertTrue(torch.isclose(poisson.l, torch.tensor(1.0)))

        poisson = Poisson.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Poisson])]
        )
        self.assertTrue(torch.isclose(poisson.l, torch.tensor(1.0)))

        poisson = Poisson.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)])]
        )
        self.assertTrue(torch.isclose(poisson.l, torch.tensor(1.5)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Poisson.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Poisson.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Poisson.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Poisson))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Poisson,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.Poisson])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        poisson = AutoLeaf(
            [FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)])]
        )
        self.assertTrue(isinstance(poisson, Poisson))
        self.assertTrue(torch.isclose(poisson.l, torch.tensor(1.5)))

    def test_base_backend_conversion(self):

        l = random.randint(1, 10)

        torch_poisson = Poisson(Scope([0]), l)
        node_poisson = BasePoisson(Scope([0]), l)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_poisson.get_params()]),
                np.array([*toBase(torch_poisson).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_poisson.get_params()]),
                np.array([*toTorch(node_poisson).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
