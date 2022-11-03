from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure.spn import SumNode
from spflow.torch.inference import log_likelihood
from spflow.torch.learning import em
from ...structure.spn.nodes.dummy_node import DummyLeaf, log_likelihood, em
import torch
import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = DummyLeaf(Scope([0]), loc=2.0)
        l2 = DummyLeaf(Scope([0]), loc=-2.0)
        sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(
            np.vstack(
                [
                    np.random.normal(2.0, 0.2, size=(10000, 1)),
                    np.random.normal(-2.0, 0.2, size=(20000, 1)),
                ]
            )
        )

        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(sum_node, data, dispatch_ctx=dispatch_ctx)
        for module_ll in dispatch_ctx.cache["log_likelihood"].values():
            if module_ll.requires_grad:
                module_ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(sum_node, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(
            torch.allclose(
                sum_node.weights,
                torch.tensor([1.0 / 3.0, 2.0 / 3.0]),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    def test_em_mixture_of_hypergeometrics(self):
        pass


if __name__ == "__main__":
    unittest.main()
