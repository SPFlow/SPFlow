"""
Created on March 07, 2019

@author: Alejandro Molina
"""

import numpy as np
from scipy.special import logsumexp
from spn.structure.leaves.parametric.Parametric import Gaussian, Bernoulli

from spn.algorithms.EM import add_node_em_update


def bernoulli_em_update(node, node_lls=None, node_gradients=None, root_lls=None, data=None, update_p=True, **kwargs):
    if not update_p:
        return

    X = data[:, node.scope[0]]

    p = (node_gradients - root_lls) + node_lls

    lse = logsumexp(p)

    wl = p - lse
    paramlse = np.exp(logsumexp(wl, b=X))

    assert not np.isnan(paramlse)

    node.p = min(max(paramlse, 0), 1)


def gaussian_em_update(
    node, node_lls=None, node_gradients=None, root_lls=None, data=None, update_mean=True, update_std=True, **kwargs
):
    p = (node_gradients - root_lls) + node_lls
    lse = logsumexp(p)
    w = np.exp(p - lse)
    X = data[:, node.scope[0]]

    mean = np.sum(w * X)  # can't be done with logsumexp, as it might be negative

    if update_mean:
        node.mean = mean

    if update_std:
        dev = np.power(X - mean, 2)
        node.stdev = np.sqrt(np.sum(w * dev))


def add_parametric_EM_support():
    add_node_em_update(Gaussian, gaussian_em_update)
    add_node_em_update(Bernoulli, bernoulli_em_update)
