"""
Created on March 21, 2018
@author: Alejandro Molina
"""
from spn.io.Text import add_str_to_spn, add_node_to_str
from collections import OrderedDict
import inspect
import re
import numpy as np

from spn.structure.leaves.parametric.Parametric import Parametric, Categorical, MultivariateGaussian
import logging

logger = logging.getLogger(__name__)


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
        except BaseException:
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
        except BaseException:
            try:
                val = float(v)
            except BaseException:
                val = list(map(float, v.children))
        init_params[str(p)] = val
    node = obj_type(**init_params)

    feature = str(tree.children[0])

    if features is not None:
        node.scope.append(features.index(feature))
    else:
        node.scope.append(int(feature[1:]))

    return node


def mvg_to_str(node, feature_names=None, node_to_str=None):


    decimals = 3

    if feature_names is None:
        fname = "V" + str(node.scope[0])
        for i in range(1, len(node.scope)):
            fname += "V" + str(node.scope[i])
    else:
        fname = feature_names[node.scope[0]]

    sigma = np.asarray(node.sigma).flatten()
    params = np.hstack((np.asarray(node.mean), sigma))

    params = np.array2string(
        params,
        separator=',',
        precision=decimals).replace(
        '\n',
        '')

    return "MultivariateGaussian(%s|prmset=%s)" % (fname, params)


def MVG_tree_to_spn(tree, features, obj_type, tree_to_spn):

    params = tree.children[1:]

    init_params = OrderedDict()
    for p, v in zip(params[::2], params[1::2]):
        val = v
        try:
            val = int(v)
        except BaseException:
            try:
                val = float(v)
            except BaseException:
                val = list(map(float, v.children))
        init_params[str(p)] = val

    params = (init_params[p])

    feature = str(tree.children[0])

    feature = re.sub("V", ",", feature)

    arr = np.fromstring(feature[1:], dtype=int, sep=',')

    scope = (list(arr))
    mean = params[:len(scope)]
    covflat = params[len(scope):]
    cov = np.reshape(covflat, (len(scope), len(scope)))
    node = MultivariateGaussian(mean, cov, scope)

    return node


def add_parametric_text_support():
    for c in Parametric.__subclasses__():
        if(c.__name__ == 'MultivariateGaussian'):
            add_node_to_str(MultivariateGaussian, mvg_to_str)
        else:
            add_node_to_str(c, parametric_to_str)

    for c in Parametric.__subclasses__():
        if(c.__name__ == 'MultivariateGaussian'):
            name = c.__name__
            add_str_to_spn(
                "parametric_" +
                name.lower(),
                MVG_tree_to_spn,
                """%s: "%s(" FNAME "|" [ PARAMNAME "=" list ] ")" """ %
                ("parametric_" +
                 name.lower(),
                    name),
                c,
            )
        else:
            name = c.__name__
            add_str_to_spn(
                "parametric_" +
                name.lower(),
                parametric_tree_to_spn,
                """%s: "%s(" FNAME "|"  [PARAMNAME "=" NUMBERS (";" PARAMNAME "=" NUMBERS )*] ")" """ %
                ("parametric_" +
                 name.lower(),
                    name),
                c,
            )

    add_str_to_spn(
        "categorical",
        categorical_tree_to_spn,
        """categorical: "Categorical(" FNAME "|" [ PARAMNAME "=" list ] ")" """,
        Categorical,
    )
