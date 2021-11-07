from spflow.base.structure.nodes.leaves.parametric import Uniform
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchUniform, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest


class TestTorchUniform(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        node_uniform = Uniform([0], start, end)
        torch_uniform = TorchUniform([0], start, end)

        # create test inputs/outputs
        data_np = np.array(
            [
                [np.nextafter(start, -np.inf)],
                [start],
                [(start + end) / 2.0],
                [end],
                [np.nextafter(end, np.inf)],
            ]
        )
        data_torch = torch.tensor(
            [
                [torch.nextafter(torch.tensor(start), -torch.tensor(float("Inf")))],
                [start],
                [(start + end) / 2.0],
                [end],
                [torch.nextafter(torch.tensor(end), torch.tensor(float("Inf")))],
            ]
        )

        log_probs = log_likelihood(SPN(), node_uniform, data_np)
        log_probs_torch = log_likelihood(torch_uniform, data_torch)

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        torch_uniform = TorchUniform([0], start, end)

        data_torch = torch.tensor(
            [
                [torch.nextafter(torch.tensor(start), -torch.tensor(float("Inf")))],
                [start],
                [(start + end) / 2.0],
                [end],
                [torch.nextafter(torch.tensor(end), torch.tensor(float("Inf")))],
            ]
        )

        log_probs_torch = log_likelihood(torch_uniform, data_torch)

        # create dummy targets
        targets_torch = torch.ones(5, 1)
        targets_torch.requires_grad = True

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_uniform.start.grad is None)
        self.assertTrue(torch_uniform.end.grad is None)

        # make sure distribution has no (learnable) parameters
        self.assertFalse(list(torch_uniform.parameters()))

    def test_base_backend_conversion(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        node_uniform = Uniform([0], start, end)
        torch_uniform = TorchUniform([0], start, end)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_uniform.get_params()]),
                np.array([*toNodes(torch_uniform).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_uniform.get_params()]),
                np.array([*toTorch(node_uniform).get_params()]),
            )
        )

    def test_initialization(self):

        start_end = random.random()

        self.assertRaises(Exception, TorchUniform, [0], start_end, start_end)
        self.assertRaises(Exception, TorchUniform, [0], start_end, np.nextafter(start_end, -1.0))
        self.assertRaises(Exception, TorchUniform, [0], np.inf, 0.0)
        self.assertRaises(Exception, TorchUniform, [0], np.nan, 0.0)
        self.assertRaises(Exception, TorchUniform, [0], 0.0, np.inf)
        self.assertRaises(Exception, TorchUniform, [0], 0.0, np.nan)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
