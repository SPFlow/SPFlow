"""
Created on October 23, 2018

@author: Nicola Di Mauro
"""
from spn.io.Text import spn_to_str_equation
from spn.io.Text import add_str_to_spn, add_node_to_str
from collections import OrderedDict
import inspect
import numpy as np

from spn.structure.leaves.cltree.CLTree import CLTree


def cltree_to_str(node, feature_names=None, node_to_str=None):
    decimals = 3

    if feature_names is None:
        fname = "V" + str(node.scope[0])
        for i in range(1, len(node.scope)):
            fname += ",V" + str(node.scope[i])
    else:
        fname = feature_names[node.scope[0]]

    factors = np.array2string(np.exp(node.log_factors), separator=",", precision=decimals).replace("\n", "")

    return "CLTREE(%s|%s)" % (fname, factors)


"""
def cltree_tree_to_spn(tree, features, obj_type, tree_to_spn):
"""


def add_cltree_text_support():
    add_node_to_str(CLTree, cltree_to_str)
