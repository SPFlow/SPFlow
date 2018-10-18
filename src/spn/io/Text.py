'''
Created on March 21, 2018

@author: Alejandro Molina
'''
import json
from collections import OrderedDict
from enum import Enum

from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, rebuild_scopes_bottom_up, assign_ids, Leaf
import numpy as np


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return str(obj)

    if isinstance(obj, Enum):
        return obj.name

    if isinstance(obj, dict):
        return dict([(str(key), json_default(val)) for key, val in obj.items()])

    if isinstance(obj, list):
        return [json_default(e) for e in obj]

    try:
        json.dumps(obj)
        return obj
    except:
        obj_dict = obj.__dict__
        values = dict([(str(key), json_default(val)) for key, val in obj_dict.items() if key[0] != "_"])
        return {obj.__class__.__name__: values}


def to_JSON(node):
    return json.dumps(node, sort_keys=True, default=json_default)


def spn_to_str_ref_graph(node, feature_names=None, node_to_str=None):
    if node_to_str is not None:
        t_node = type(node)
        if t_node in node_to_str:
            return node_to_str[t_node](node, feature_names, node_to_str)

    if isinstance(node, Leaf):
        return str(node) + " " + spn_to_str_equation(node, feature_names) + "\n"

    if isinstance(node, Product):
        pd = ", ".join(map(lambda c: c.name, node.children))
        chld_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
        chld_str = chld_str.replace("\n", "\n\t")
        return "%s ProductNode(%s){\n\t%s}\n" % (str(node), pd, chld_str)

    if isinstance(node, Sum):
        w = node.weights
        ch = node.children
        sumw = ", ".join(map(lambda i: "%s*%s" % (w[i], ch[i].name), range(len(ch))))
        child_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
        child_str = child_str.replace("\n", "\n\t")
        return "%s SumNode(%s){\n\t%s}\n" % (str(node), sumw, child_str)

    raise Exception('Node type not registered: ' + str(type(node)))


_node_to_str = {}


def add_node_to_str(node_type, lambda_func):
    _node_to_str[node_type] = lambda_func


def spn_to_str_equation(node, feature_names=None, node_to_str=_node_to_str):
    t_node = type(node)
    if t_node in node_to_str:
        return node_to_str[t_node](node, feature_names, node_to_str)

    if isinstance(node, Product):
        children_strs = map(lambda child: spn_to_str_equation(child, feature_names, node_to_str), node.children)
        return "(" + " * ".join(children_strs) + ")"

    if isinstance(node, Sum):
        def fmt_chld(w, c): return str(w) + "*(" + spn_to_str_equation(c, feature_names, node_to_str) + ")"

        children_strs = map(lambda i: fmt_chld(node.weights[i], node.children[i]), range(len(node.children)))

        return "(" + " + ".join(children_strs) + ")"

    raise Exception('Node type not registered: ' + str(type(node)))


_str_to_spn = OrderedDict()


def add_str_to_spn(name, lambda_func, grammar, obj_type):
    _str_to_spn[name] = (lambda_func, grammar, obj_type)


def str_to_spn(text, features=None, str_to_spn_lambdas=_str_to_spn):
    from lark import Lark

    ext_name = "\n".join(map(lambda s: "    | " + s, str_to_spn_lambdas.keys()))

    ext_grammar = "\n".join([s for _, s, _ in str_to_spn_lambdas.values()])

    grammar = r"""
%import common.DECIMAL -> DECIMAL
%import common.WS
%ignore WS
%import common.WORD -> WORD
%import common.DIGIT -> DIGIT
ALPHANUM: "a".."z"|"A".."Z"|DIGIT
PARAMCHARS: ALPHANUM|"_"
FNAME: ALPHANUM+
PARAMNAME: PARAMCHARS+
NUMBER: DIGIT|DECIMAL
NUMBERS: NUMBER+
list: "[" [NUMBERS ("," NUMBERS)*] "]"


?node: prodnode
    | sumnode
""" + ext_name + r"""

prodnode: "(" [node ("*" node)*] ")"
sumnode: "(" [NUMBERS "*" node ("+" NUMBERS "*" node)*] ")"

""" + ext_grammar

    parser = Lark(grammar, start='node')
    # print(grammar)
    tree = parser.parse(text)

    def tree_to_spn(tree, features):
        tnode = tree.data

        if tnode == "sumnode":
            node = Sum()
            for i in range(int(len(tree.children) / 2)):
                j = 2 * i
                w, c = tree.children[j], tree.children[j + 1]
                node.weights.append(float(w))
                node.children.append(tree_to_spn(c, features))
            return node

        if tnode == "prodnode":
            if len(tree.children) == 1:
                return tree_to_spn(tree.children[0], features)
            node = Product()
            for c in tree.children:
                node.children.append(tree_to_spn(c, features))
            return node

        if tnode in str_to_spn_lambdas:
            return str_to_spn_lambdas[tnode][0](tree, features, str_to_spn_lambdas[tnode][2], tree_to_spn)

        raise Exception('Node type not registered: ' + tnode)

    spn = tree_to_spn(tree, features)

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)
    valid, err = is_valid(spn)
    assert valid, err
    assign_ids(spn)
    return spn
