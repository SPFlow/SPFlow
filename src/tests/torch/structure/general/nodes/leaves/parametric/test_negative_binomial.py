import random
import unittest

import numpy as np
import torch

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import NegativeBinomial as BaseNegativeBinomial
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure import AutoLeaf, marginalize, toBase, toTorch
from spflow.torch.structure.spn import NegativeBinomial


class TestNegativeBinomial(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Negative Binomial distribution: p in (0,1], n > 0

        # p = 1
        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)
        # p = 0
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, 0.0)
        # p < 0 and p > 1
        self.assertRaises(
            Exception,
            NegativeBinomial,
            Scope([0]),
            1,
            torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)),
        )
        self.assertRaises(
            Exception,
            NegativeBinomial,
            [0],
            1,
            torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)),
        )
        # p = +-inf and p = nan
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.inf)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, -np.inf)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.nan)

        # n = 0
        NegativeBinomial(Scope([0]), 0.0, 1.0)
        # n < 0
        self.assertRaises(
            Exception,
            NegativeBinomial,
            Scope([0]),
            torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)),
            1.0,
        )
        # n = inf and n = nan
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), np.nan, 1.0)

        # invalid scopes
        self.assertRaises(Exception, NegativeBinomial, Scope([]), 1, 1.0)
        self.assertRaises(Exception, NegativeBinomial, Scope([0, 1]), 1, 1.0)
        self.assertRaises(Exception, NegativeBinomial, Scope([0], [1]), 1, 1.0)

    def test_structural_marginalization(self):

        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)

        self.assertTrue(marginalize(negative_binomial, [1]) is not None)
        self.assertTrue(marginalize(negative_binomial, [0]) is None)

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(
            NegativeBinomial.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # Bernoulli feature type instance
        self.assertTrue(
            NegativeBinomial.accepts(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.NegativeBinomial(n=3)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            NegativeBinomial.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # conditional scope
        self.assertFalse(
            NegativeBinomial.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)]
                    )
                ]
            )
        )

        # multivariate signature
        self.assertFalse(
            NegativeBinomial.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [
                            FeatureTypes.NegativeBinomial(n=3),
                            FeatureTypes.Binomial(n=3),
                        ],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        negative_binomial = NegativeBinomial.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])]
        )
        self.assertTrue(torch.isclose(negative_binomial.n, torch.tensor(3)))
        self.assertTrue(torch.isclose(negative_binomial.p, torch.tensor(0.5)))

        negative_binomial = NegativeBinomial.from_signatures(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.NegativeBinomial(n=3, p=0.75)]
                )
            ]
        )
        self.assertTrue(torch.isclose(negative_binomial.n, torch.tensor(3)))
        self.assertTrue(torch.isclose(negative_binomial.p, torch.tensor(0.75)))

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            NegativeBinomial.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            NegativeBinomial.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            NegativeBinomial.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            NegativeBinomial.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(NegativeBinomial))

        # make sure leaf is correctly inferred
        self.assertEqual(
            NegativeBinomial,
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.NegativeBinomial(n=3)]
                    )
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        negative_binomial = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.NegativeBinomial(n=3, p=0.75)]
                )
            ]
        )
        self.assertTrue(isinstance(negative_binomial, NegativeBinomial))
        self.assertTrue(torch.isclose(negative_binomial.n, torch.tensor(3)))
        self.assertTrue(torch.isclose(negative_binomial.p, torch.tensor(0.75)))

    def test_base_backend_conversion(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_negative_binomial = NegativeBinomial(Scope([0]), n, p)
        node_negative_binomial = BaseNegativeBinomial(Scope([0]), n, p)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_negative_binomial.get_params()]),
                np.array([*toBase(torch_negative_binomial).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_negative_binomial.get_params()]),
                np.array([*toTorch(node_negative_binomial).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
