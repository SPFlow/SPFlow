import random
import unittest

import numpy as np
import torch

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import CondCategorical as BaseCondCategorical
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import CondCategorical


class TestCondCategorical(unittest.TestCase):

    def test_likelihood_module_cond_f(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]
        cond_f = lambda data: {"k": k, "p": p}

        condCategorical = CondCategorical(Scope([0], [1]), cond_f=cond_f)

        data = torch.tensor([[i] for i in range(k)])
        targets = torch.tensor([[p_i] for p_i in p])

        probs = likelihood(condCategorical, data)
        log_probs = log_likelihood(condCategorical, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))


    def test_likelihood_args_p(self):

        condCategorical = CondCategorical(Scope([0], [1]))

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[condCategorical] = {"k": k, "p": p}

        data = torch.tensor([[i] for i in range(k)])
        targets = torch.tensor([[p_i] for p_i in p])

        probs = likelihood(condCategorical, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(condCategorical, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))


    def test_likelihood_args_cond_f(self):

        condCategorical = CondCategorical(Scope([0], [1]))

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]
        cond_f = lambda data: {"k": k, "p": p}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[condCategorical] = {"cond_f": cond_f}

        data = torch.tensor([[i] for i in range(k)])
        targets= torch.tensor([[p_i] for p_i in p])

        probs = likelihood(condCategorical, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(condCategorical, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))


    def test_inference(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]
        cond_f = lambda data: {"k": k, "p": p}
        
        torch_categorical = CondCategorical(Scope([0], [1]), cond_f=cond_f)
        base_categorical = BaseCondCategorical(Scope([0], [1]), cond_f=cond_f)

        data = np.random.randint(0, k, (10, 1))

        log_probs = log_likelihood(base_categorical, data)
        log_probs_torch = log_likelihood(torch_categorical, torch.tensor(data))

        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))


    def test_gradient_computation(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = torch.tensor([p_i / sum(p) for p_i in p], requires_grad=True)
        cond_f = lambda data: {"k": k, "p": p}
        
        torch_categorical = CondCategorical(Scope([0], [1]), cond_f=cond_f)

        data = np.random.randint(0, k, (10, 1))

        log_probs_torch = log_likelihood(torch_categorical, torch.tensor(data))

        targets_torch = torch.ones(10, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(p.grad is not None)


    def test_likelihood(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]
        cond_f = lambda data: {"k": k, "p": p}
        
        condCategorical = CondCategorical(Scope([0], [1]), cond_f=cond_f)

        data = torch.tensor([[i] for i in range(k)])
        targets = torch.tensor([[p_i] for p_i in p])

        probs = likelihood(condCategorical, data)
        log_probs = log_likelihood(condCategorical, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))


    def test_likelihood_marginalization(self):
        
        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]
        cond_f = lambda data: {"k": k, "p": p}
        
        condCategorical = CondCategorical(Scope([0], [1]), cond_f=cond_f)
        data = torch.tensor([[float("nan")]])

        probs = likelihood(condCategorical, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    
    def test_support(self):

        # support for Categorical distribution: integers in {0, 1, ..., k-1}

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i / sum(p) for p_i in p]
        cond_f = lambda data: {"k": k, "p": p}


        condCategorical = CondCategorical(Scope([0], [1]), cond_f=cond_f)

        # check inf
        self.assertRaises(ValueError, log_likelihood, condCategorical, torch.tensor([[-float("inf")]]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, torch.tensor([[float("inf")]]))        
        
        # check valid integers in valid range
        data = torch.tensor([[0], [1], [2], [3]])
        log_likelihood(condCategorical, data)

        # check valid integers outside valid range
        self.assertRaises(ValueError, log_likelihood, condCategorical, torch.tensor([[-1]]))
        self.assertRaises(ValueError, log_likelihood, condCategorical, torch.tensor([[k]]))

        # check invalid values     
        self.assertRaises(
            ValueError,
            log_likelihood,
            condCategorical,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            condCategorical,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            condCategorical,
            torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            condCategorical,
            torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))]]),
        )
        self.assertRaises(ValueError, log_likelihood, condCategorical, torch.tensor([[0.5]]))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
   
        

