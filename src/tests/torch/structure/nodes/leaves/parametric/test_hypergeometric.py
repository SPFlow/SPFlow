from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric as BaseHypergeometric
from spflow.base.inference.nodes.leaves.parametric.hypergeometric import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric, toBase, toTorch
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.hypergeometric import log_likelihood
from spflow.torch.inference.module import likelihood
#from spflow.torch.sampling import sample

import torch
import numpy as np

import random
import unittest


class TestHypergeometric(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Hypergeometric distribution: N in N U {0}, M in {0,...,N}, n in {0,...,N}, p in [0,1] TODO

        # N = 0
        Hypergeometric(Scope([0]), 0, 0, 0)
        # N < 0
        self.assertRaises(Exception, Hypergeometric, Scope([0]), -1, 1, 1)
        # N = inf and N = nan
        self.assertRaises(Exception, Hypergeometric, Scope([0]), np.inf, 1, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), np.nan, 1, 1)
        # N float
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1.5, 1, 1)

        # M < 0 and M > N
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, -1, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 2, 1)
        # 0 <= M <= N
        for i in range(4):
            Hypergeometric(Scope([0]), 3, i, 0)
        # M = inf and M = nan
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, np.inf, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, np.nan, 1)
        # M float
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 0.5, 1)

        # n < 0 and n > N
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, -1)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, 2)
        # 0 <= n <= N
        for i in range(4):
            Hypergeometric(Scope([0]), 3, 0, i)
        # n = inf and n = nan
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, np.inf)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, np.nan)
        # n float
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, 0.5)

        # invalid scopes
        self.assertRaises(Exception, Hypergeometric, Scope([]), 1, 1, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0, 1]), 1, 1, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0],[1]), 1, 1, 1)
    
    def test_structural_marginalization(self):

        hypergeometric = Hypergeometric(Scope([0]), 0, 0, 0)

        self.assertTrue(marginalize(hypergeometric, [1]) is not None)
        self.assertTrue(marginalize(hypergeometric, [0]) is None)

    def test_base_backend_conversion(self):

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = Hypergeometric(Scope([0]), N, M, n)
        node_hypergeometric = BaseHypergeometric(Scope([0]), N, M, n)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_hypergeometric.get_params()]),
                np.array([*toBase(torch_hypergeometric).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_hypergeometric.get_params()]),
                np.array([*toTorch(node_hypergeometric).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
