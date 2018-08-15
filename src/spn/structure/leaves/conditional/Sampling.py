from spn.algorithms.Sampling import add_node_sampling
from spn.structure.leaves.conditional.Conditional import Conditional, Conditional_Bernoulli, Conditional_Gaussian, Conditional_Poisson

import numpy as np

from spn.structure.leaves.conditional.utils import get_scipy_obj_params


def sample_conditional_node(node, obs, rand_gen):   # n_samples -> obs
    assert isinstance(node, Conditional)
    assert len(obs) > 0

    X = None
    if isinstance(node, Conditional_Bernoulli) or isinstance(node, Conditional_Gaussian) or isinstance(node, Conditional_Poisson):

        scipy_obj, params = get_scipy_obj_params(node, obs)

        assert len(params) == obs.shape[0]

        # X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
        X = [scipy_obj.rvs(size=1, random_state=rand_gen, **param) for param in params]

        assert X.shape[0] == obs.shape[0]

    else:
        raise Exception('Node type unknown: ' + str(type(node)))

    return X


def add_conditional_sampling_support():
    add_node_sampling(Conditional_Bernoulli, sample_conditional_node)
    add_node_sampling(Conditional_Gaussian, sample_conditional_node)
    add_node_sampling(Conditional_Poisson, sample_conditional_node)

