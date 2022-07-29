from spflow.base.sampling.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric import Uniform
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchUniform, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood
from spflow.torch.sampling import sample

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

        log_probs = log_likelihood(node_uniform, data_np, SPN())
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

        # Valid parameters for Uniform distribution: a<b

        # start = end
        start_end = random.random()
        self.assertRaises(Exception, TorchUniform, [0], start_end, start_end)
        # start > end
        self.assertRaises(
            Exception,
            TorchUniform,
            [0],
            start_end,
            torch.nextafter(torch.tensor(start_end), torch.tensor(-1.0)),
        )
        # start = +-inf and start = nan
        self.assertRaises(Exception, TorchUniform, [0], np.inf, 0.0)
        self.assertRaises(Exception, TorchUniform, [0], -np.inf, 0.0)
        self.assertRaises(Exception, TorchUniform, [0], np.nan, 0.0)
        # end = +-inf and end = nan
        self.assertRaises(Exception, TorchUniform, [0], 0.0, np.inf)
        self.assertRaises(Exception, TorchUniform, [0], 0.0, -np.inf)
        self.assertRaises(Exception, TorchUniform, [0], 0.0, np.nan)

        # invalid scope lengths
        self.assertRaises(Exception, TorchUniform, [], 0.0, 1.0)
        self.assertRaises(Exception, TorchUniform, [0, 1], 0.0, 1.0)

    def test_support(self):

        # Support for Uniform distribution: floats [a,b] or (-inf,inf)

        # ----- with support outside the interval -----
        uniform = TorchUniform([0], 1.0, 2.0, support_outside=True)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, uniform, torch.tensor([[-float("inf")]]))
        self.assertRaises(ValueError, log_likelihood, uniform, torch.tensor([[float("inf")]]))

        # check valid floats in [start, end]
        log_likelihood(uniform, torch.tensor([[1.0]]))
        log_likelihood(uniform, torch.tensor([[1.5]]))
        log_likelihood(uniform, torch.tensor([[2.0]]))

        # check valid floats outside [start, end]
        log_likelihood(
            uniform, torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(-1.0))]])
        )
        log_likelihood(
            uniform, torch.tensor([[torch.nextafter(torch.tensor(2.0), torch.tensor(3.0))]])
        )

        # ----- without support outside the interval -----
        uniform = TorchUniform([0], 1.0, 2.0, support_outside=False)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, uniform, torch.tensor([[-float("inf")]]))
        self.assertRaises(ValueError, log_likelihood, uniform, torch.tensor([[float("inf")]]))

        # check valid floats in [start, end]
        log_likelihood(uniform, torch.tensor([[1.0]]))
        log_likelihood(uniform, torch.tensor([[1.5]]))
        log_likelihood(uniform, torch.tensor([[2.0]]))

        # check invalid floats outside
        self.assertRaises(
            ValueError,
            log_likelihood,
            uniform,
            torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(-1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            uniform,
            torch.tensor([[torch.nextafter(torch.tensor(2.0), torch.tensor(3.0))]]),
        )

    def test_marginalization(self):

        uniform = TorchUniform([0], 1.0, 2.0)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(uniform, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_sampling(self):

        # ----- a = -1.0, b = 2.5 -----

        uniform = TorchUniform([0], -1.0, 2.5)
        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(uniform, data, ll_cache={}, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))

        samples = sample(uniform, 1000)
        self.assertTrue(all((samples >= -1.0) & (samples <= 2.5)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
