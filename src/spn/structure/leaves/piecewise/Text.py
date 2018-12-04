"""
Created on March 21, 2018

@author: Alejandro Molina
"""
from spn.io.Text import add_str_to_spn, add_node_to_str

from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


def piecewise_to_str(node, feature_names=None, node_to_str=None):
    import numpy as np

    decimals = 4
    if feature_names is None:
        fname = "V" + str(node.scope[0])
    else:
        fname = feature_names[node.scope[0]]

    x_range = np.array2string(np.array(node.x_range), precision=decimals, separator=",")
    y_range = np.array2string(
        np.array(node.y_range), precision=decimals, separator=",", formatter={"float_kind": lambda x: "%.10f" % x}
    )

    return "PiecewiseLinear(%s|%s;%s)" % (fname, x_range, y_range)


def piecewise_tree_to_spn(tree, features, obj_type, tree_to_spn):
    node = PiecewiseLinear(list(map(float, tree.children[1].children)), list(map(float, tree.children[2].children)))

    feature = str(tree.children[0])

    node.scope.append(features.index(feature))

    return node


def add_piecewise_text_support():
    add_node_to_str(PiecewiseLinear, piecewise_to_str)

    add_str_to_spn(
        "pwl",
        piecewise_tree_to_spn,
        """
                   pwl: "PiecewiseLinear(" FNAME "|" list ";" list ")"  """,
        None,
    )
