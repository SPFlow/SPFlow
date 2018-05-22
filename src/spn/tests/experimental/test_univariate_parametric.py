'''
Created on April 10, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Inference import parametric_likelihood
from spn.structure.leaves.Parametric import ParametricScipy

import numpy as np

if __name__ == '__main__':
    node = ParametricScipy('norm', loc=0.0, scale=1.0)

    data = np.vstack((np.asarray([1.5, 0.5]), np.asarray([0.5, 0.5]),
                      np.asarray([0.7, 0.5]), np.asarray([0.5, 0.7])))

    ll = parametric_likelihood(node, data[:,0], log_space=True)

    assert np.isclose(ll, -2.04393853)

    print(ll)