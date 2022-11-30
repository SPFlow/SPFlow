import random
import unittest

import numpy as np
import torch

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Uniform as BaseUniform
from spflow.meta.data import Scope
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import Uniform


class TestUniform(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        node_uniform = BaseUniform(Scope([0]), start, end)
        torch_uniform = Uniform(Scope([0]), start, end)

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

        log_probs = log_likelihood(node_uniform, data_np)
        log_probs_torch = log_likelihood(torch_uniform, data_torch)

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        torch_uniform = Uniform(Scope([0]), start, end)

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

    def test_likelihood_marginalization(self):

        uniform = Uniform(Scope([0]), 1.0, 2.0)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(uniform, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Uniform distribution: floats [a,b] or (-inf,inf)

        # ----- with support outside the interval -----
        uniform = Uniform(Scope([0]), 1.0, 2.0, support_outside=True)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, uniform, torch.tensor([[-float("inf")]]))
        self.assertRaises(ValueError, log_likelihood, uniform, torch.tensor([[float("inf")]]))

        # check valid floats in [start, end]
        log_likelihood(uniform, torch.tensor([[1.0]]))
        log_likelihood(uniform, torch.tensor([[1.5]]))
        log_likelihood(uniform, torch.tensor([[2.0]]))

        # check valid floats outside [start, end]
        log_likelihood(
            uniform,
            torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(-1.0))]]),
        )
        log_likelihood(
            uniform,
            torch.tensor([[torch.nextafter(torch.tensor(2.0), torch.tensor(3.0))]]),
        )

        # ----- without support outside the interval -----
        uniform = Uniform(Scope([0]), 1.0, 2.0, support_outside=False)

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


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
