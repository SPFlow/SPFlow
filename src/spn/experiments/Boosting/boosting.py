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

    spn = 0.5 * Gaussian(0.0, 1.0, scope=0) + 0.5 * Gaussian(2.0, 1.0, scope=0)

    data = np.array((1, 2, 3)).reshape((-1, 1))

    print(np.exp(log_likelihood(spn, data)))
