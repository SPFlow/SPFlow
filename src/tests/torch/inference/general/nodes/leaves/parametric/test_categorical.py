import random
import unittest

import numpy as np
import torch

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Categorical as BaseCategorical
from spflow.meta.data import Scope
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import Categorical


class TestCategorical(unittest.TestCase):
    def test_inference(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]

        torch_categorical = Categorical(Scope([0]), k=k, p=p)
        base_categorical = BaseCategorical(Scope([0]), k=k, p=p)

        data = np.random.randint(0, k, (10, 1))

        log_probs = log_likelihood(base_categorical, data)
        log_probs_torch = log_likelihood(torch_categorical, torch.tensor(data))

        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))
        

    def test_gradient_computation(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]

        torch_categorical = Categorical(Scope([0]), k=k, p=p)
        
        data = np.random.randint(0, k, (10, 1))

        log_probs_torch = log_likelihood(torch_categorical, torch.tensor(data))

        # dummy targets
        targets_torch = torch.ones(10, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_categorical.p_aux.grad is not None)

        p_aux_orig = torch_categorical.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_categorical.parameters(), lr=1)
        optimizer.step()

        self.assertTrue(torch.allclose(p_aux_orig - torch_categorical.p_aux.grad, torch_categorical.p_aux))

        self.assertTrue(torch.allclose(torch_categorical.p, torch_categorical.dist.probs, atol=1e-1, rtol=1e-2))


    def test_gradient_optimization(self):

        torch.manual_seed(0)

        torch_categorical = Categorical(Scope([0]), k=2, p=[0.3, 0.7])

        # dummy data
        p_target = torch.tensor([0.8, 0.2])
        data = torch.distributions.Categorical(p_target).sample((10000, 1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_categorical.parameters(), lr=0.5, momentum=0.5)

        for _ in range(40):
            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_categorical, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        p = torch_categorical.p.detach().numpy()
        torch_categorical.p = p / sum(p)

        self.assertTrue(torch.allclose(torch_categorical.p, torch.tensor([0.8, 0.2]), atol=2e-2, rtol=1e-2))


    def test_likelihood_p(self):

        categorical = Categorical(Scope([0]), k=2, p=[0.3, 0.7])

        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.3], [0.7]])

        probs = likelihood(categorical, data)
        log_probs = log_likelihood(categorical, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))


    def test_likelihood_marginalization(self):

        categorical = Categorical(Scope([0]), k=2, p=[0.3, 0.7])
        data = torch.tensor([[float("nan")]])

        probs = likelihood(categorical, data)
        
        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))


    def test_support(self):

        # Support for Categorical distribution: integers in {0, 1, ..., k-1}

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]

        categorical = Categorical(Scope([0]), k=k, p=p)

        self.assertRaises(ValueError, log_likelihood, categorical, torch.tensor([[-float("inf")]]))
        self.assertRaises(ValueError, log_likelihood, categorical, torch.tensor([[float("inf")]]))

        # valid integers inside valid range
        log_likelihood(categorical, torch.tensor([[0], [1]]))

        # valid integers outside valid range
        self.assertRaises(ValueError, log_likelihood, categorical, torch.tensor([[-1]]))
        self.assertRaises(ValueError, log_likelihood, categorical, torch.tensor([[k]]))

        # invalid values
        self.assertRaises(
            ValueError,
            log_likelihood,
            categorical,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            categorical,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            categorical,
            torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            categorical,
            torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))]]),
        )
        self.assertRaises(ValueError, log_likelihood, categorical, torch.tensor([[0.5]]))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
    

