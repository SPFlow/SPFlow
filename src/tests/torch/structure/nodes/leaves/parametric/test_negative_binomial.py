from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.negative_binomial import (
    NegativeBinomial as BaseNegativeBinomial,
)
from spflow.base.inference.nodes.leaves.parametric.negative_binomial import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.negative_binomial import (
    NegativeBinomial,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.negative_binomial import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


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
