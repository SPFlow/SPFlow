import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.modules.module import log_likelihood
from spflow.torch.learning import em
from spflow.structure.spn import SumNode
from spflow.utils import Tensor
from spflow.tensor import ops as tle

from ....structure.general.node.dummy_node import DummyLeaf, em, log_likelihood

tc = unittest.TestCase()


def test_em_step(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l1 = DummyLeaf(Scope([0]), loc=2.0)
    l2 = DummyLeaf(Scope([0]), loc=-2.0)
    sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

    data = tl.tensor(
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

    tc.assertTrue(
        np.allclose(
            tle.toNumpy(sum_node.weights),
            tl.tensor([1.0 / 3.0, 2.0 / 3.0]),
            atol=1e-2,
            rtol=1e-2,
        )
    )


def test_em_mixture_of_hypergeometrics(do_for_all_backends):
    pass


if __name__ == "__main__":
    unittest.main()
