import random
import unittest

import torch

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_uniform import UniformLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_uniform import Uniform


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

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

        dummy_data = torch.tensor([[0.5, -0.3], [0.9, 0.21], [0.5, 0.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

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

        self.assertTrue(torch_uniform.start.grad is None)
        self.assertTrue(torch_uniform.end.grad is None)

        # make sure distribution has no (learnable) parameters
        #self.assertFalse(list(torch_uniform.parameters()))

    def test_likelihood_marginalization(self):

        uniform = UniformLayer(
            scope=[Scope([0]), Scope([1])],
            start=0.0,
            end=random.random() + 1e-7,
        )
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(uniform, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
