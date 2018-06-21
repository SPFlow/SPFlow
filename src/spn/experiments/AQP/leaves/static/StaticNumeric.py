'''
Created on June 21, 2018

@author: Moritz
'''

from spn.structure.Base import Leaf

class StaticNumeric(Leaf):
    def __init__(self, val, scope=None):
        Leaf.__init__(self, scope=scope)
        self.val = val

    @property
    def mode(self):
        return self.val


def create_static_leaf(val, scope):
    assert len(scope) == 1, "scope of univariate Piecewise for more than one variable?"
    return StaticNumeric(val, scope)
