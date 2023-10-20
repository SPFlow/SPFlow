import random
import unittest

import numpy as np
import pytest
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import log_likelihood
from spflow.torch.learning import (
    em,
    expectation_maximization,
    maximum_likelihood_estimation,
)
from spflow.tensorly.structure.spn import MultivariateGaussianLayer
from spflow.tensorly.structure.spn import ProductNode
from spflow.tensorly.utils.helper_functions import tl_toNumpy
from spflow.torch.structure.general.layers.leaves.parametric.multivariate_gaussian import updateBackend

tc = unittest.TestCase()

def test_mle(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    layer = MultivariateGaussianLayer(scope=[Scope([0, 1]), Scope([2, 3])])

    # simulate data
    data = np.hstack(
        [
            np.random.multivariate_normal(
                mean=tl.tensor([-1.7, 0.3]),
                cov=tl.tensor([[1.0, 0.25], [0.25, 0.5]]),
                size=(20000,),
            ),
            np.random.multivariate_normal(
                mean=tl.tensor([0.5, 0.2]),
                cov=tl.tensor([[1.3, -0.7], [-0.7, 1.0]]),
                size=(20000,),
            ),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    tc.assertTrue(
        np.allclose(
            tl_toNumpy(tl_toNumpy(layer.mean[0])),
            tl.tensor([-1.7, 0.3]),
            atol=1e-2,
            rtol=1e-2,
        )
    )
    tc.assertTrue(
        np.allclose(
            tl_toNumpy(tl_toNumpy(layer.mean[1])),
            tl.tensor([0.5, 0.2]),
            atol=1e-2,
            rtol=1e-2,
        )
    )
    tc.assertTrue(
        np.allclose(
            tl_toNumpy(tl_toNumpy(layer.cov[0])),
            tl.tensor([[[1.0, 0.25], [0.25, 0.5]]]),
            atol=1e-2,
            rtol=1e-2,
        )
    )
    tc.assertTrue(
        np.allclose(
            tl_toNumpy(tl_toNumpy(layer.cov[1])),
            tl.tensor([[[1.3, -0.7], [-0.7, 1.0]]]),
            atol=1e-2,
            rtol=1e-2,
        )
    )

def test_em_step(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    layer = MultivariateGaussianLayer([Scope([0, 1]), Scope([2, 3])])
    data = tl.tensor(
        np.hstack(
            [
                np.random.multivariate_normal(
                    np.array([-0.5, 0.5]),
                    np.array([[1.2, 0.0], [0.0, 0.8]]),
                    size=(15000,),
                ),
                np.random.multivariate_normal(
                    np.array([0.5, -0.5]),
                    np.array([[0.8, 0.0], [0.0, 1.2]]),
                    size=(15000,),
                ),
            ]
        )
    )
    dispatch_ctx = DispatchContext()

    # compute gradients of log-likelihoods w.r.t. module log-likelihoods
    ll = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)

    for module_ll in dispatch_ctx.cache["log_likelihood"].values():
        module_ll.retain_grad()

    ll.sum().backward()

    # perform an em step
    em(layer, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(layer.mean[0]), tl.tensor([-0.5, 0.5]), atol=1e-2, rtol=1e-1))
    tc.assertTrue(np.allclose(tl_toNumpy(layer.mean[1]), tl.tensor([0.5, -0.5]), atol=1e-2, rtol=1e-1))
    tc.assertTrue(
        np.allclose(
            tl_toNumpy(layer.cov[0]),
            tl.tensor([[1.2, 0.0], [0.0, 0.8]]),
            atol=1e-2,
            rtol=1e-1,
        )
    )
    tc.assertTrue(
        np.allclose(
            tl_toNumpy(layer.cov[1]),
            tl.tensor([[0.8, 0.0], [0.0, 1.2]]),
            atol=1e-2,
            rtol=1e-1,
        )
    )

def test_update_backend(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    layer = MultivariateGaussianLayer(scope=[Scope([0, 1]), Scope([2, 3])])

    # simulate data
    data = np.hstack(
        [
            np.random.multivariate_normal(
                mean=tl.tensor([-1.7, 0.3]),
                cov=tl.tensor([[1.0, 0.25], [0.25, 0.5]]),
                size=(20000,),
            ),
            np.random.multivariate_normal(
                mean=tl.tensor([0.5, 0.2]),
                cov=tl.tensor([[1.3, -0.7], [-0.7, 1.0]]),
                size=(20000,),
            ),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    mean = tl_toNumpy(np.array(layer.get_params()[0]))
    cov = tl_toNumpy(np.array(layer.get_params()[1]))

    layer = MultivariateGaussianLayer(scope=[Scope([0, 1]), Scope([2, 3])])
    prod_node = ProductNode([layer])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    mean_em = tl_toNumpy(np.array(layer.get_params()[0]))
    cov_em = tl_toNumpy(np.array(layer.get_params()[1]))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer = MultivariateGaussianLayer(scope=[Scope([0, 1]), Scope([2, 3])])
            layer_updated = updateBackend(layer)
            maximum_likelihood_estimation(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            mean_updated = tl_toNumpy(np.array(layer_updated.get_params()[0]))
            cov_updated = tl_toNumpy(np.array(layer_updated.get_params()[1]))
            tc.assertTrue(np.allclose(mean, mean_updated, atol=1e-2, rtol=1e-1))
            tc.assertTrue(np.allclose(cov, cov_updated, atol=1e-2, rtol=1e-1))

            layer = MultivariateGaussianLayer(scope=[Scope([0, 1]), Scope([2, 3])])
            layer_updated = updateBackend(layer)
            prod_node = ProductNode([layer_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                mean_em_updated = tl_toNumpy(np.array(layer_updated.get_params()[0]))
                cov_em_updated = tl_toNumpy(np.array(layer_updated.get_params()[1]))
                tc.assertTrue(np.allclose(mean_em, mean_em_updated, atol=1e-2, rtol=1e-1))
                tc.assertTrue(np.allclose(cov_em, cov_em_updated, atol=1e-2, rtol=1e-1))


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    unittest.main()
