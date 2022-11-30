from spflow.meta.data import Scope, FeatureTypes, FeatureContext
from spflow.torch.structure import AutoLeaf
from spflow.torch.structure.spn import Bernoulli
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.inference import log_likelihood
from spflow.base.structure.spn import Bernoulli as BaseBernoulli
from spflow.base.inference import log_likelihood

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
        self.assertTrue(
            Bernoulli.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # Bernoulli feature type class
        self.assertTrue(
            Bernoulli.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])]
            )
        )

        # Bernoulli feature type instance
        self.assertTrue(
            Bernoulli.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.5)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Bernoulli.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # conditional scope
        self.assertFalse(
            Bernoulli.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Bernoulli.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        bernoulli = Bernoulli.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
        )
        self.assertTrue(torch.isclose(bernoulli.p, torch.tensor(0.5)))

        bernoulli = Bernoulli.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])]
        )
        self.assertTrue(torch.isclose(bernoulli.p, torch.tensor(0.5)))

        bernoulli = Bernoulli.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])]
        )
        self.assertTrue(torch.isclose(bernoulli.p, torch.tensor(0.75)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Bernoulli.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Bernoulli.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Bernoulli.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Bernoulli))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Bernoulli,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        bernoulli = AutoLeaf(
            [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])]
        )
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
