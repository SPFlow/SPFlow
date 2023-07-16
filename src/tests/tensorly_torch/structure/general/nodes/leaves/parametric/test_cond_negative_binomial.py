import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_negative_binomial import (
    CondNegativeBinomial as BaseCondNegativeBinomial,
)
from spflow.torch.structure.general.nodes.leaves.parametric.cond_negative_binomial import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.spn import CondNegativeBinomial as TorchCondNegativeBinomial
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_negative_binomial import CondNegativeBinomial
from spflow.torch.structure.general.nodes.leaves.parametric.cond_negative_binomial import (
    #CondNegativeBinomial,
    toBase,
    toTorch,
)
from spflow.tensorly.structure.spn.nodes.sum_node import marginalize


class TestNegativeBinomial(unittest.TestCase):
    def test_initialization(self):

        binomial = CondNegativeBinomial(Scope([0], [1]), n=1)
        self.assertTrue(binomial.cond_f is None)
        binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=lambda x: {"p": 0.5})
        self.assertTrue(isinstance(binomial.cond_f, Callable))

        # n = 0
        CondNegativeBinomial(Scope([0], [1]), 0.0)
        # n < 0
        self.assertRaises(
            Exception,
            CondNegativeBinomial,
            Scope([0]),
            torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)),
        )
        # n = inf and n = nan
        self.assertRaises(Exception, CondNegativeBinomial, Scope([0]), np.inf)
        self.assertRaises(Exception, CondNegativeBinomial, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, CondNegativeBinomial, Scope([]), 1)
        self.assertRaises(Exception, CondNegativeBinomial, Scope([0, 1], [2]), 1)
        self.assertRaises(Exception, CondNegativeBinomial, Scope([0]), 1)

    def test_retrieve_params(self):

        # Valid parameters for Negative Binomial distribution: p in (0,1], n > 0
        negative_binomial = CondNegativeBinomial(Scope([0], [1]), 1)

        # p = 1
        negative_binomial.set_cond_f(lambda data: {"p": 1.0})
        self.assertTrue(negative_binomial.retrieve_params(np.array([[1.0]]), DispatchContext()) == 1.0)
        # p = 0
        negative_binomial.set_cond_f(lambda data: {"p": 0.0})
        self.assertRaises(
            Exception,
            negative_binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # p < 0 and p > 1
        negative_binomial.set_cond_f(lambda data: {"p": torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))})
        self.assertRaises(
            Exception,
            negative_binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        negative_binomial.set_cond_f(lambda data: {"p": torch.nextafter(torch.tensor(0.0), -torch.tensor(1.0))})
        self.assertRaises(
            Exception,
            negative_binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # p = +-inf and p = nan
        negative_binomial.set_cond_f(lambda data: {"p": torch.tensor(float("inf"))})
        self.assertRaises(
            Exception,
            negative_binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        negative_binomial.set_cond_f(lambda data: {"p": -torch.tensor(float("inf"))})
        self.assertRaises(
            Exception,
            negative_binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        negative_binomial.set_cond_f(lambda data: {"p": torch.tensor(float("nan"))})
        self.assertRaises(
            Exception,
            negative_binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(CondNegativeBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # Bernoulli feature type instance
        self.assertTrue(
            CondNegativeBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)])])
        )

        # invalid feature type
        self.assertFalse(CondNegativeBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

        # non-conditional scope
        self.assertFalse(
            CondNegativeBinomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])])
        )

        # multivariate signature
        self.assertFalse(
            CondNegativeBinomial.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [
                            FeatureTypes.NegativeBinomial(n=3),
                            FeatureTypes.Binomial(n=3),
                        ],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondNegativeBinomial.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)])])
        CondNegativeBinomial.from_signatures(
            [
                FeatureContext(
                    Scope([0], [1]),
                    [FeatureTypes.NegativeBinomial(n=3, p=0.75)],
                )
            ]
        )

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            CondNegativeBinomial.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondNegativeBinomial.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondNegativeBinomial.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondNegativeBinomial.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondNegativeBinomial))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondNegativeBinomial,
            AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        negative_binomial = AutoLeaf(
            [
                FeatureContext(
                    Scope([0], [1]),
                    [FeatureTypes.NegativeBinomial(n=3, p=0.75)],
                )
            ]
        )
        self.assertTrue(isinstance(negative_binomial, TorchCondNegativeBinomial))
        self.assertEqual(negative_binomial.n, torch.tensor(3))

    def test_structural_marginalization(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), 1)

        self.assertTrue(marginalize(negative_binomial, [1]) is not None)
        self.assertTrue(marginalize(negative_binomial, [0]) is None)

    def test_base_backend_conversion(self):

        n = random.randint(2, 10)

        torch_negative_binomial = CondNegativeBinomial(Scope([0], [1]), n)
        node_negative_binomial = BaseCondNegativeBinomial(Scope([0], [1]), n)

        # check conversion from torch to python
        self.assertTrue(np.all(torch_negative_binomial.scopes_out == toBase(torch_negative_binomial).scopes_out))
        # check conversion from python to torch
        self.assertTrue(np.all(node_negative_binomial.scopes_out == toTorch(node_negative_binomial).scopes_out))

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        n = random.randint(2, 10)

        cond_negative_binomial = CondNegativeBinomial(Scope([0], [1]), n)
        for backend in backends:
            tl.set_backend(backend)
            cond_negative_binomial_updated = updateBackend(cond_negative_binomial)

            # check conversion from torch to python
            self.assertTrue(np.all(cond_negative_binomial.scopes_out == cond_negative_binomial_updated.scopes_out))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
