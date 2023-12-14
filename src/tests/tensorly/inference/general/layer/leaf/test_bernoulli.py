import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layer.leaf.general_bernoulli import BernoulliLayer
from spflow.tensorly.structure.general.node.leaf.general_bernoulli import Bernoulli
from spflow.torch.structure.general.layer.leaf.bernoulli import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):

    layer = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.5, 0.9])

    nodes = [
        Bernoulli(Scope([0]), p=0.2),
        Bernoulli(Scope([1]), p=0.5),
        Bernoulli(Scope([0]), p=0.9),
    ]

    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_layer_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    p = [random.random(), random.random()]

    bernoulli = BernoulliLayer(scope=[Scope([0]), Scope([1])], p=p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 2, (3, 2))

    log_probs_torch = log_likelihood(bernoulli, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(bernoulli.p_aux.grad is not None)

    p_aux_orig = bernoulli.p_aux.detach().clone()

    optimizer = torch.optim.SGD(bernoulli.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(np.allclose(tl_toNumpy(p_aux_orig - bernoulli.p_aux.grad), tl_toNumpy(bernoulli.p_aux)))

    # verify that distribution parameters match parameters
    tc.assertTrue(np.allclose(tl_toNumpy(bernoulli.p), tl_toNumpy(bernoulli.dist().probs)))

def test_gradient_optimization(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # initialize distribution
    torch_bernoulli = BernoulliLayer(scope=[Scope([0]), Scope([1])], p=[0.3, 0.7])

    # create dummy data
    p_target = tl.tensor([0.8, 0.2])
    data = torch.bernoulli(p_target.unsqueeze(0).repeat((100000, 1)))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_bernoulli.parameters(), lr=0.5, momentum=0.5)

    # perform optimization (possibly overfitting)
    for i in range(50):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_bernoulli, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    context = tl.context(torch_bernoulli.p)

    tc.assertTrue(np.allclose(tl_toNumpy(torch_bernoulli.p), tl_toNumpy(p_target), atol=1e-2, rtol=1e-2))

def test_likelihood_marginalization(do_for_all_backends):

    bernoulli = BernoulliLayer(scope=[Scope([0]), Scope([1])], p=random.random())
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(bernoulli, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    layer = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.5, 0.9])

    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float32)

    layer_ll = log_likelihood(layer, dummy_data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            bernoulli_updated = updateBackend(layer)
            log_probs_updated = log_likelihood(bernoulli_updated, tl.tensor(dummy_data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    layer = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.5, 0.9])
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float32)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    layer.to_dtype(tl.float64)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float64)
    layer_ll_up = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    layer = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.5, 0.9])
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]])
    layer_ll = log_likelihood(layer, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    layer.to_device(cuda)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], device=cuda)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
