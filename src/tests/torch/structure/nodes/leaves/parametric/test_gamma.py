from spflow.meta.data import Scope, FeatureTypes, FeatureContext
from spflow.torch.structure import AutoLeaf
from spflow.torch.structure.spn import Gamma
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.inference import log_likelihood
from spflow.base.structure.spn import Gamma as BaseGamma
from spflow.base.inference import log_likelihood

import torch
import numpy as np

import random
import unittest


class TestGamma(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        # alpha > 0
        Gamma(
            Scope([0]),
            torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)),
            1.0,
        )
        # alpha = 0
        self.assertRaises(Exception, Gamma, Scope([0]), 0.0, 1.0)
        # alpha < 0
        self.assertRaises(
            Exception, Gamma, Scope([0]), np.nextafter(0.0, -1.0), 1.0
        )
        # alpha = inf and alpha = nan
        self.assertRaises(Exception, Gamma, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0]), np.nan, 1.0)

        # beta > 0
        Gamma(
            Scope([0]),
            1.0,
            torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)),
        )
        # beta = 0
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, 0.0)
        # beta < 0
        self.assertRaises(
            Exception, Gamma, Scope([0]), 1.0, np.nextafter(0.0, -1.0)
        )
        # beta = inf and beta = non
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.inf)
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Gamma, Scope([]), 1.0, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0, 1]), 1.0, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0], [1]), 1.0, 1.0)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            Gamma.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # Gamma feature type class
        self.assertTrue(
            Gamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Gamma])])
        )

        # Gamma feature type instance
        self.assertTrue(
            Gamma.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.0, 1.0)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Gamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])])
        )

        # conditional scope
        self.assertFalse(
            Gamma.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Gamma.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        gamma = Gamma.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
        )
        self.assertTrue(torch.isclose(gamma.alpha, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(gamma.beta, torch.tensor(1.0)))

        gamma = Gamma.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Gamma])]
        )
        self.assertTrue(torch.isclose(gamma.alpha, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(gamma.beta, torch.tensor(1.0)))

        gamma = Gamma.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.5, 0.5)])]
        )
        self.assertTrue(torch.isclose(gamma.alpha, torch.tensor(1.5)))
        self.assertTrue(torch.isclose(gamma.beta, torch.tensor(0.5)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Gamma.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Gamma.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Gamma.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Gamma))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Gamma,
            AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Gamma])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gamma = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.Gamma(alpha=1.5, beta=0.5)]
                )
            ]
        )
        self.assertTrue(isinstance(gamma, Gamma))
        self.assertTrue(torch.isclose(gamma.alpha, torch.tensor(1.5)))
        self.assertTrue(torch.isclose(gamma.beta, torch.tensor(0.5)))

    def test_structural_marginalization(self):

        gamma = Gamma(Scope([0]), 1.0, 1.0)

        self.assertTrue(marginalize(gamma, [1]) is not None)
        self.assertTrue(marginalize(gamma, [0]) is None)

    def test_base_backend_conversion(self):

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = Gamma(Scope([0]), alpha, beta)
        node_gamma = BaseGamma(Scope([0]), alpha, beta)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_gamma.get_params()]),
                np.array([*toBase(torch_gamma).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_gamma.get_params()]),
                np.array([*toTorch(node_gamma).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
