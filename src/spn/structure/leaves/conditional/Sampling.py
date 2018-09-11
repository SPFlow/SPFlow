from spn.algorithms.Sampling import add_node_sampling
from spn.structure.leaves.conditional.Conditional import Conditional, Conditional_Bernoulli, Conditional_Gaussian, \
    Conditional_Poisson

import numpy as np

from spn.structure.leaves.conditional.utils import get_scipy_obj_params


def sample_conditional_node(node, n_samples, data, rand_gen):  # n_samples -> obs
    assert len(data) > 0

    if not isinstance(node, (Conditional_Bernoulli, Conditional_Gaussian, Conditional_Poisson)):
        raise Exception('Node type unknown: ' + str(type(node)))

    scipy_obj, params = get_scipy_obj_params(node, data[:, -node.evidence_size:])

    if isinstance(node, Conditional_Poisson):
        params['mu'] = np.clip(params['mu'], 0., 256.)  # todo tmp test
    try:
        X = scipy_obj.rvs(size=data.shape[0], random_state=rand_gen, **params)
    except Exception:
        print("node", node, node.weights)
        print("params", params, np.shape(params))
        print("data shape", np.shape(data))
        print("input shape", np.shape(data[:, -node.evidence_size:]))
        0/0

    assert X.shape[0] == data.shape[0]

    return X




def add_conditional_sampling_support():
    add_node_sampling(Conditional_Bernoulli, sample_conditional_node)
    add_node_sampling(Conditional_Gaussian, sample_conditional_node)
    add_node_sampling(Conditional_Poisson, sample_conditional_node)
