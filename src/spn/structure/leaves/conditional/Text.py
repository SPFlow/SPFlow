'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from spn.io.Text import spn_to_str_equation
from spn.io.Text import add_str_to_spn, add_node_to_str
from collections import OrderedDict
import inspect
import numpy as np


from spn.structure.leaves.conditional.Conditional import Conditional

def conditional_to_str(node, feature_names=None, node_to_str=None):
    decimals = 4
    if feature_names is None:
        fname = "V" + str(node.scope[0])
    else:
        fname = feature_names[node.scope[0]]

    weights = np.array2string(np.array(node.weights), precision=decimals, separator=',',
                                formatter={'float_kind': lambda x: "%.10f" % x})

    return "Conditional(%s|%s)" % (fname, weights)


def conditional_to_spn(tree, features, obj_type, tree_to_spn):
    node = Conditional(list(map(float, tree.children[1].children)), list(map(float, tree.children[2].children)))

    feature = str(tree.children[0])

    node.scope.append(features.index(feature))

    return node


def add_histogram_text_support():
    add_node_to_str(Conditional, conditional_to_str)

    add_str_to_spn("conditional", conditional_to_spn,
                   """
                   conditional: "conditional(" FNAME "|" list ";" list ")"  """, None)
