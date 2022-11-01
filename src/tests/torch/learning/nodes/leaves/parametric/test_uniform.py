from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.torch.structure.spn.nodes.node import SPNSumNode, SPNProductNode
from spflow.torch.inference.spn.nodes.node import log_likelihood
from spflow.torch.learning.spn.nodes.node import em
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.torch.learning.nodes.leaves.parametric.uniform import (
    maximum_likelihood_estimation,
    em,
)
from spflow.torch.inference.nodes.leaves.parametric.uniform import (
    log_likelihood,
)
from spflow.torch.learning.expectation_maximization.expectation_maximization import (
    expectation_maximization,
)

import torch
import numpy as np
import random
import unittest


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_mle(self):

        leaf = Uniform(Scope([0]), start=0.0, end=1.0)

        # simulate data
        data = torch.tensor([[0.5]])

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(
            torch.all(
                torch.tensor([leaf.start, leaf.end]) == torch.tensor([0.0, 1.0])
            )
        )

    def test_mle_invalid_support(self):

        leaf = Uniform(Scope([0]), start=1.0, end=3.0, support_outside=False)

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("inf")]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[0.0]]),
            bias_correction=True,
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)

        leaf = Uniform(Scope([0]), start=-3.0, end=4.5)
        data = torch.rand((100, 1)) * 7.5 - 3.0
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.requires_grad = True
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(
            torch.all(
                torch.tensor([leaf.start, leaf.end])
                == torch.tensor([-3.0, 4.5])
            )
        )

    def test_em_product_of_uniforms(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Uniform(Scope([0]), start=-1.0, end=3.0)
        l2 = Uniform(Scope([1]), start=2.0, end=5.0)
        prod_node = SPNProductNode([l1, l2])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.rand(15000, 1) * 4.0 - 1.0,
                    np.random.rand(15000, 1) * 3.0 + 2.0,
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(
            torch.all(
                torch.tensor([l1.start, l1.end]) == torch.tensor([-1.0, 3.0])
            )
        )
        self.assertTrue(
            torch.all(
                torch.tensor([l2.start, l2.end]) == torch.tensor([2.0, 5.0])
            )
        )

    def test_em_sum_of_uniforms(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Uniform(Scope([0]), start=-1.0, end=3.0)
        l2 = Uniform(Scope([0]), start=-1.0, end=3.0)
        sum_node = SPNSumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(np.random.rand(15000, 1) * 3.0 + 2.0)

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(
            torch.all(
                torch.tensor([l1.start, l1.end]) == torch.tensor([-1.0, 3.0])
            )
        )
        self.assertTrue(
            torch.all(
                torch.tensor([l2.start, l2.end]) == torch.tensor([-1.0, 3.0])
            )
        )


if __name__ == "__main__":
    unittest.main()
