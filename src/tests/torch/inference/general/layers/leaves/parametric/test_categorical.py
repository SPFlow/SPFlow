import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure.spn import Categorical, CategoricalLayer


class TestCategorical(unittest.TestCase):

    def test_layer_likelihood(self):

        scopes = [Scope([0]), Scope([1]), Scope([0])]
        ks = [2, 2, 2]
        ps = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]
        layer = CategoricalLayer(scope=scopes, k=ks, p=ps)

        nodes = [Categorical(scope=scope, k=k, p=p) for scope, k, p in zip(scopes, ks, ps)]

        data = torch.tensor([[1, 0], [0, 0], [1, 1]])

        layer_ll = log_likelihood(layer, data)
        nodes_ll = torch.concat([log_likelihood(node, data) for node in nodes], axis=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    
    def test_layer_gradient_computation(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i/sum(p) for p_i in p]

        torch_categorical = CategoricalLayer(scope=[Scope([0]), Scope([1])], k=k, p=p)

        data = np.random.randint(0, k, (10, 2))

        log_probs_torch = log_likelihood(torch_categorical, torch.tensor(data))

        targets_torch = torch.ones(10, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_categorical.p_aux.grad is not None)

        p_aux_orig = torch_categorical.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_categorical.parameters(), lr=1)
        optimizer.step()

        self.assertTrue(torch.allclose(p_aux_orig - torch_categorical.p_aux.grad, torch_categorical.p_aux))
        self.assertTrue(torch.allclose(torch_categorical.p, torch_categorical.dist().probs, atol=1e-1, rtol=1e-2))


    def test_gradient_optimization(self):

        torch.manual_seed(0)

        torch_categorical = CategoricalLayer(scope=[Scope([0]), Scope([1])], k=2, p=[[0.5, 0.5], [0.3, 0.7]])
        
        p_target = torch.tensor([[0.8, 0.2], [0.5, 0.5]])
        data = torch.distributions.Categorical(p_target).sample(torch.Size((10000,)))
        print(data)

        optimizer = torch.optim.SGD(torch_categorical.parameters(), lr=0.5, momentum=0.5)

        for i in range(50):

            optimizer.zero_grad()

            nll = -log_likelihood(torch_categorical, data).mean()
            nll.backward()

            optimizer.step()

        p = torch_categorical.p.detach().numpy()
        torch_categorical.p = torch.tensor(np.array([pp / np.sum(pp) for pp in p]))

        self.assertTrue(torch.allclose(torch_categorical.p, p_target, atol=2e-2, rtol=1e-2))

    
    def test_likelihood_marginalization(self):

        categorical = CategoricalLayer(scope=[Scope([0]), Scope([1])], k=2, p=[0.3, 0.7])
        data = torch.tensor([[float("nan"), float("nan")]])

        probs = log_likelihood(categorical, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()