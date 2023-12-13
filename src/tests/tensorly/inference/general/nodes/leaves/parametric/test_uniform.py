import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import Uniform
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.uniform import Uniform as BaseUniform
from spflow.torch.structure.general.nodes.leaves.parametric.uniform import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    start = random.random()
    end = start + 1e-7 + random.random()

    node_uniform = BaseUniform(Scope([0]), start, end)
    uniform = Uniform(Scope([0]), start, end)

    # create test inputs/outputs
    data_np = np.array(
        [
            [np.nextafter(start, -np.inf)],
            [start],
            [(start + end) / 2.0],
            [end],
            [np.nextafter(end, np.inf)],
        ]
        , dtype=np.float32
    )
    data_torch = tl.tensor(
        data_np
    , dtype=tl.float32)

    probs = log_likelihood(node_uniform, data_np)
    probs_torch = log_likelihood(uniform, data_torch)

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(probs, probs_torch, atol=0.01, rtol=0.01))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    start = random.random()
    end = start + 1e-7 + random.random()

    uniform = Uniform(Scope([0]), start, end)

    data_torch = tl.tensor(
        [
            [np.nextafter(tl.tensor(start), -tl.tensor(float("Inf")))],
            [start],
            [(start + end) / 2.0],
            [end],
            [np.nextafter(tl.tensor(end), tl.tensor(float("Inf")))],
        ]
    , dtype=tl.float32)

    log_probs_torch = log_likelihood(uniform, data_torch)

    # create dummy targets
    targets_torch = torch.ones(5, 1)
    targets_torch.requires_grad = True

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(uniform.start.grad is None)
    tc.assertTrue(uniform.end.grad is None)

    # make sure distribution has no (learnable) parameters
    #tc.assertFalse(list(uniform.parameters()))

def test_likelihood_marginalization(do_for_all_backends):

    uniform = Uniform(Scope([0]), 1.0, 2.0)
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(uniform, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    # Support for Uniform distribution: floats [a,b] or (-inf,inf)

    # ----- with support outside the interval -----
    uniform = Uniform(Scope([0]), 1.0, 2.0, support_outside=True)

    # check infinite values
    tc.assertRaises(ValueError, log_likelihood, uniform, tl.tensor([[-float("inf")]]))
    tc.assertRaises(ValueError, log_likelihood, uniform, tl.tensor([[float("inf")]]))

    # check valid floats in [start, end]
    log_likelihood(uniform, tl.tensor([[1.0]], dtype=tl.float32))
    log_likelihood(uniform, tl.tensor([[1.5]], dtype=tl.float32))
    log_likelihood(uniform, tl.tensor([[2.0]], dtype=tl.float32))

    # check valid floats outside [start, end]
    log_likelihood(
        uniform,
        tl.tensor([[np.nextafter(tl.tensor(1.0), tl.tensor(-1.0))]]),
    )
    log_likelihood(
        uniform,
        tl.tensor([[np.nextafter(tl.tensor(2.0), tl.tensor(3.0))]]),
    )

    # ----- without support outside the interval -----
    uniform = Uniform(Scope([0]), 1.0, 2.0, support_outside=False)

    # check infinite values
    tc.assertRaises(ValueError, log_likelihood, uniform, tl.tensor([[-float("inf")]]))
    tc.assertRaises(ValueError, log_likelihood, uniform, tl.tensor([[float("inf")]]))

    # check valid floats in [start, end]
    log_likelihood(uniform, tl.tensor([[1.0]], dtype=tl.float32))
    log_likelihood(uniform, tl.tensor([[1.5]], dtype=tl.float32))
    log_likelihood(uniform, tl.tensor([[2.0]], dtype=tl.float32))

    # check invalid floats outside
    tc.assertRaises(
        ValueError,
        log_likelihood,
        uniform,
        tl.tensor([[np.nextafter(tl.tensor(1.0), tl.tensor(-1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        uniform,
        tl.tensor([[np.nextafter(tl.tensor(2.0), tl.tensor(3.0))]]),
    )

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    start = random.random()
    end = start + 1e-7 + random.random()

    uniform = Uniform(Scope([0]), start, end)

    # create test inputs/outputs
    """
    data_np = np.array(
        [
            [np.nextafter(start, -np.inf)],
            [start],
            [(start + end) / 2.0],
            [end],
            [np.nextafter(end, np.inf)],
        ]
    )
    """
    data_torch = tl.tensor(
        [
            [np.nextafter(tl.tensor(start), -tl.tensor(float("Inf")))],
            [start],
            [(start + end) / 2.0],
            [end],
            [np.nextafter(tl.tensor(end), tl.tensor(float("Inf")))],
        ]
    , dtype=tl.float32)

    log_probs = log_likelihood(uniform, data_torch)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            uniform_updated = updateBackend(uniform)
            log_probs_updated = log_likelihood(uniform_updated, tl.tensor(data_torch, dtype=tl.float32))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(log_probs, log_probs_updated, atol=0.001, rtol=0.001))

def test_change_dtype(do_for_all_backends):
    start = random.random()
    end = start + 1e-7 + random.random()

    node = Uniform(Scope([0]), start, end)
    dummy_data = tl.tensor(np.random.rand(3, 1), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor(np.random.rand(3, 1), dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    start = random.random()
    end = start + 1e-7 + random.random()

    node = Uniform(Scope([0]), start, end)
    dummy_data = tl.tensor(np.random.rand(3, 1), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.random.rand(3, 1), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
