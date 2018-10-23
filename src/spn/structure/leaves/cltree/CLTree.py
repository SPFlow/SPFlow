'''
Created on Ocotber 19, 2018
@author: Nicola Di Mauro
'''

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import Type

class CLTree(Leaf):
    def __init__(self, scope=None):
        self._type = Type.BINARY
        Leaf.__init__(self, scope=scope)

    @property
    def type(self):
        return self._type
        
    @property
    def params(self):
        raise Exception("Not Implemented")
    
   
def create_cltree_leaf(data, ds_context, scope):
    from spn.structure.leaves.cltree.MLE import update_cltree_parameters_mle
    
    node = CLTree(scope)
    update_cltree_parameters_mle(node, data)
    
    return node
