'''
Created on Ocotber 19, 2018
@author: Nicola Di Mauro
'''

from spn.structure.Base import Leaf


class CLTree(Leaf):
    def __init__(self, scope=None):
        Leaf.__init__(self, scope=scope)

    @property
    def type(self):
        raise Exception("Not Implemented")

    @property
    def params(self):
        raise Exception("Not Implemented")


def create_cltree_leaf(data, ds_context, scope):
    from spn.structure.leaves.cltree.MLE import update_cltree_parameters_mle

    node = CLTree(scope)
    update_parametric_parameters_mle(node, data)

    return node
