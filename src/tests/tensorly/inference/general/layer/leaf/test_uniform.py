import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layer.leaf.general_uniform import UniformLayer
from spflow.tensorly.structure.general.node.leaf.general_uniform import Uniform
from spflow.torch.structure.general.layer.leaf.uniform import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):

    layer = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -1.0, 0.3],
        end=[1.0, 0.3, 0.97],
    )

    nodes = [
        Uniform(Scope([0]), start=0.2, end=1.0, support_outside=True),
        Uniform(Scope([1]), start=-1.0, end=0.3, support_outside=True),
        Uniform(Scope([0]), start=0.3, end=0.97, support_outside=True),
    ]

    dummy_data = tl.tensor([[0.5, -0.3], [0.9, 0.21], [0.5, 0.0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    start = torch.tensor([random.random(), random.random()])
    end = start + 1e-7 + torch.tensor([random.random(), random.random()])

    torch_uniform = UniformLayer(scope=[Scope([0]), Scope([1])], start=start, end=end)

    data_torch = torch.stack(
        [
            torch.nextafter(start, -torch.tensor(float("Inf"))),
            start,
            (start + end) / 2.0,
            end,
            torch.nextafter(end, torch.tensor(float("Inf"))),
        ]
    )

    log_probs_torch = log_likelihood(torch_uniform, data_torch)

    # create dummy targets
    targets_torch = torch.ones(5, 2)
    targets_torch.requires_grad = True

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_uniform.start.grad is None)
    tc.assertTrue(torch_uniform.end.grad is None)

    # make sure distribution has no (learnable) parameters
    #tc.assertFalse(list(torch_uniform.parameters()))

def test_likelihood_marginalization(do_for_all_backends):

    uniform = UniformLayer(
        scope=[Scope([0]), Scope([1])],
        start=0.0,
        end=random.random() + 1e-7,
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(uniform, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    layer = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -1.0, 0.3],
        end=[1.0, 0.3, 0.97],
    )

    dummy_data = tl.tensor([[0.5, -0.3], [0.9, 0.21], [0.5, 0.0]])

    layer_ll = log_likelihood(layer, dummy_data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(layer)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    layer = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -1.0, 0.3],
        end=[1.0, 0.3, 0.97],
    )
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
    layer = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -1.0, 0.3],
        end=[1.0, 0.3, 0.97],
    )
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
