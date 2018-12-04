"""
Created on June 21, 2018

@author: Moritz
"""
import numpy as np

from spn.structure.Base import Leaf


class IdentityNumeric(Leaf):
    def __init__(self, vals, mean, scope=None):
        Leaf.__init__(self, scope=scope)
        self.vals = vals
        self.mean = mean


def create_identity_leaf(data, scope):
    assert len(scope) == 1, "scope for more than one variable?"

    vals = np.sort(data[:, 0])
    mean = np.nanmean(vals)

    return IdentityNumeric(vals, mean, scope)
