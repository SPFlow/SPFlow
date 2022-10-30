from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.binomial import (
    Binomial as BaseBinomial,
)
from spflow.base.inference.nodes.leaves.parametric.binomial import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.binomial import (
    Binomial,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.binomial import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestBinomial(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Binomial distribution: p in [0,1], n in N U {0}

        # p = 0
        binomial = Binomial(Scope([0]), 1, 0.0)
        # p = 1
        binomial = Binomial(Scope([0]), 1, 1.0)
        # p < 0 and p > 1
        self.assertRaises(
            Exception,
            Binomial,
            Scope([0]),
            1,
            torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)),
        )
        self.assertRaises(
            Exception,
            Binomial,
            Scope([0]),
            1,
            torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)),
        )
        # p = inf and p = nan
        self.assertRaises(Exception, Binomial, Scope([0]), 1, np.inf)
        self.assertRaises(Exception, Binomial, Scope([0]), 1, np.nan)

        # n = 0
        binomial = Binomial(Scope([0]), 0, 0.5)
        # n < 0
        self.assertRaises(Exception, Binomial, Scope([0]), -1, 0.5)
        # n float
        self.assertRaises(Exception, Binomial, Scope([0]), 0.5, 0.5)
        # n = inf and n = nan
        self.assertRaises(Exception, Binomial, Scope([0]), np.inf, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0]), np.nan, 0.5)

        # invalid scopes
        self.assertRaises(Exception, Binomial, Scope([]), 1, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0, 1]), 1, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0], [1]), 1, 0.5)

    def test_structural_marginalization(self):

        binomial = Binomial(Scope([0]), 1, 0.5)

        self.assertTrue(marginalize(binomial, [1]) is not None)
        self.assertTrue(marginalize(binomial, [0]) is None)

    def test_base_backend_conversion(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = Binomial(Scope([0]), n, p)
        node_binomial = BaseBinomial(Scope([0]), n, p)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_binomial.get_params()]),
                np.array([*toBase(torch_binomial).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_binomial.get_params()]),
                np.array([*toTorch(node_binomial).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
