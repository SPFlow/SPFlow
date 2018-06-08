'''
Created on June 07, 2018

@author: Alejandro Molina
'''

import torch as torch
from torch.distributions import Normal

from spn.io.Text import spn_to_str_equation
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.Text import add_parametric_text_support

if __name__ == '__main__':
    add_parametric_text_support()

    torch.manual_seed(1234)
    m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    L = Gaussian(mean=0.0, stdev=1.0, scope=[0]) * Gaussian(mean=1.0, stdev=1.0, scope=[1])
    R = Gaussian(mean=2.0, stdev=1.0, scope=[0]) * Gaussian(mean=3.0, stdev=1.0, scope=[1])
    spn = 0.4 * L + 0.6 * R

    print(spn_to_str_equation(spn))


    def to_pytorch(node):

        if isinstance(node, Gaussian):
            node._torch_mean = torch.tensor([0.0])
            node._torch_stdev = torch.tensor([0.0])
            return Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    print(m.log_prob(0.2))
