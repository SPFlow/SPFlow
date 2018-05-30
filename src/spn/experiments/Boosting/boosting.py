'''
Created on May 14, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Inference import log_likelihood
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian
import numpy as np

if __name__ == '__main__':
    add_parametric_inference_support()

    spnL = 0.5 * Gaussian(0.0, 1.0, scope=0) + 0.5 * Gaussian(2.0, 1.0, scope=0)
    spnR = 0.5 * Gaussian(0.0, 1.0, scope=1) + 0.5 * Gaussian(2.0, 1.0, scope=1)

    spn = spnL * spnR

    data = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 2))
    print(data)

    print(np.exp(log_likelihood(spn, data)))
