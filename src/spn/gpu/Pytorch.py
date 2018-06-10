'''
Created on June 07, 2018

@author: Alejandro Molina
'''

import torch
from torch.distributions import Normal

from spn.io.Text import spn_to_str_equation
from spn.structure.Base import Product, Sum
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.Text import add_parametric_text_support
import numpy as np

if __name__ == '__main__':
    add_parametric_text_support()

    torch.manual_seed(1234)

    L = Gaussian(mean=0.0, stdev=1.0, scope=[0]) * Gaussian(mean=1.0, stdev=1.0, scope=[1])
    R = Gaussian(mean=2.0, stdev=1.0, scope=[0]) * Gaussian(mean=3.0, stdev=1.0, scope=[1])
    spn = 0.4 * L + 0.6 * R

    print(spn_to_str_equation(spn))

    a = torch.autograd.Variable(torch.tensor(0), requires_grad=True)
    b = torch.autograd.Variable(torch.tensor(2), requires_grad=True)
    c += a + b

    print("first", c)

    c.backward(retain_graph=True)
    c = a + b
    a[0] = 5
    c.backward(retain_graph=True)
    print("second", c)

    0 / 0
    np.random.seed(17)
    data = np.random.randn(5, 2)
    print(data)


    def to_pytorch(node, data):

        if isinstance(node, Gaussian):
            node._torch_mean = torch.tensor([0.0])
            node._torch_stdev = torch.tensor([0.0])
            return Normal(torch.tensor([0.0]), torch.tensor([1.0])).log_prob(data[:, node.scope[0]])

        ll_children = [to_pytorch(n, data) for n in node.children]

        if isinstance(node, Product):
            return torch.sum(ll_children)

        if isinstance(node, Sum):
            return torch.log(torch.bmm(ll_children, node.weights))


    print(to_pytorch(spn, data))
