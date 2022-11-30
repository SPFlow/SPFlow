import random
import unittest

import numpy as np
import torch

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import Gaussian as BaseGaussian
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure import AutoLeaf, marginalize, toBase, toTorch
from spflow.torch.structure.spn import Gaussian


class TestGaussian(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Exponential distribution: mean in R, std > 0

        mean = random.random()

        # mean = inf and mean = nan
        self.assertRaises(Exception, Gaussian, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), -np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), np.nan, 1.0)

        # std = 0 and std < 0
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, 0.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, np.nextafter(0.0, -1.0))
        # std = inf and std = nan
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, np.inf)
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Gaussian, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0], [1]), 0.0, 1.0)

    def test_structural_marginalization(self):

        gaussian = Gaussian(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(gaussian, [1]) is not None)
        self.assertTrue(marginalize(gaussian, [0]) is None)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(Gaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # Gaussian feature type class
        self.assertTrue(Gaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Gaussian])]))

        # Gaussian feature type instance
        self.assertTrue(Gaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Gaussian(0.0, 1.0)])]))

        # invalid feature type
        self.assertFalse(Gaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

        # conditional scope
        self.assertFalse(Gaussian.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

        # multivariate signature
        self.assertFalse(
            Gaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        gaussian = Gaussian.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Continuous])])
        self.assertTrue(torch.isclose(gaussian.mean, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(gaussian.std, torch.tensor(1.0)))

        gaussian = Gaussian.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Gaussian])])
        self.assertTrue(torch.isclose(gaussian.mean, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(gaussian.std, torch.tensor(1.0)))

        gaussian = Gaussian.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Gaussian(-1.0, 1.5)])])
        self.assertTrue(torch.isclose(gaussian.mean, torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(gaussian.std, torch.tensor(1.5)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Gaussian.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Gaussian.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Gaussian.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Gaussian))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Gaussian,
            AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Gaussian])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gaussian = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Gaussian(mean=-1.0, std=0.5)])])
        self.assertTrue(isinstance(gaussian, Gaussian))
        self.assertTrue(torch.isclose(gaussian.mean, torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(gaussian.std, torch.tensor(0.5)))

    def test_base_backend_conversion(self):

        mean = random.random()
        std = random.random() + 1e-7  # offset by small number to avoid zero

        torch_gaussian = Gaussian(Scope([0]), mean, std)
        node_gaussian = BaseGaussian(Scope([0]), mean, std)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_gaussian.get_params()]),
                np.array([*toBase(torch_gaussian).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_gaussian.get_params()]),
                np.array([*toTorch(node_gaussian).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
