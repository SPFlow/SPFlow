'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from spn.io.Text import add_str_to_spn, add_node_to_str
from collections import OrderedDict
import inspect

from spn.structure.leaves.parametric.Parametric import Parametric, Categorical


def parametric_to_str(node, feature_names=None, node_to_str=None):
    if feature_names is None:
        fname = "V" + str(node.scope[0])
    else:
        fname = feature_names[node.scope[0]]

    args = OrderedDict()

    for i, arg in enumerate(inspect.getfullargspec(node.__init__).args[1:]):
        if arg == "scope":
            continue
        args[arg] = getattr(node, arg)

    return "%s(%s|%s)" % (node.__class__.__name__, fname,
                          ";".join(["%s=%s" % (k, v) for k, v in args.items()]))


def parametric_tree_to_spn(tree, features, obj_type, tree_to_spn):
    params = tree.children[1:]

    init_params = OrderedDict()
    for p, v in zip(params[::2], params[1::2]):
        val = v
        try:
            val = int(v)
        except:
            val = float(v)
        init_params[str(p)] = val
    node = obj_type(**init_params)

    feature = str(tree.children[0])

    if features is not None:
        node.scope.append(features.index(feature))
    else:
        node.scope.append(int(feature[1:]))

    return node


def categorical_tree_to_spn(tree, features, obj_type, tree_to_spn):
    params = tree.children[1:]

    init_params = OrderedDict()
    for p, v in zip(params[::2], params[1::2]):
        val = v
        try:
            val = int(v)
        except:
            try:
                val = float(v)
            except:
                val = list(map(float, v.children))
        init_params[str(p)] = val
    node = obj_type(**init_params)

    feature = str(tree.children[0])

    if features is not None:
        node.scope.append(features.index(feature))
    else:
        node.scope.append(int(feature[1:]))

    return node


def add_parametric_text_support():
    for c in Parametric.__subclasses__():
        add_node_to_str(c, parametric_to_str)

    for c in Parametric.__subclasses__():
        name = c.__name__
        add_str_to_spn("parametric_" + name.lower(), parametric_tree_to_spn,
                       """%s: "%s(" FNAME "|"  [PARAMNAME "=" NUMBERS (";" PARAMNAME "=" NUMBERS )*] ")" """ %
                       ("parametric_" + name.lower(), name), c)

    add_str_to_spn("categorical", categorical_tree_to_spn,
                   """categorical: "Categorical(" FNAME "|" [ PARAMNAME "=" list ] ")" """,
                   Categorical)
