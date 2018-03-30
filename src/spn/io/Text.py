'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Pruning import prune
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, rebuild_scopes_bottom_up, assign_ids
from spn.structure.leaves.Histograms import Histogram


def histogram_to_str(node, feature_names=None, decimals=4):
    import numpy as np

    if feature_names is None:
        fname = "V" + str(node.scope[0])
    else:
        fname = feature_names[node.scope[0]]

    breaks = np.array2string(np.array(node.breaks), precision=decimals, separator=',')
    densities = np.array2string(np.array(node.densities), precision=decimals, separator=',',
                                formatter={'float_kind': lambda x: "%.10f" % x})

    return "Histogram(%s|%s;%s)" % (fname, breaks, densities)


def to_JSON(node):
    import json

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return {obj.__class__.__name__: obj.__dict__}

    return json.dumps(node, default=dumper)


def spn_to_str_ref_graph(node, feature_names=None, node_to_str=None):
    if node_to_str is not None:
        t_node = type(node)
        if t_node in node_to_str:
            return node_to_str[t_node](node, feature_names, node_to_str)

    if isinstance(node, Histogram):
        return node.name + " " + histogram_to_str(node, feature_names) + "\n"

    if isinstance(node, Product):
        pd = ", ".join(map(lambda c: c.name, node.children))
        chld_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
        chld_str = chld_str.replace("\n", "\n\t")
        return "%s ProductNode(%s){\n\t%s}\n" % (node.name, pd, chld_str)

    if isinstance(node, Sum):
        w = node.weights
        ch = node.children
        sumw = ", ".join(map(lambda i: "%s*%s" % (w[i], ch[i].name), range(len(ch))))
        child_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
        child_str = child_str.replace("\n", "\n\t")
        return "%s SumNode(%s){\n\t%s}\n" % (node.name, sumw, child_str)

    raise Exception('Node type not registered: ' + str(type(node)))


def spn_to_str_equation(node, feature_names=None, node_to_str=None):
    if node_to_str is not None:
        t_node = type(node)
        if t_node in node_to_str:
            return node_to_str[t_node](node, feature_names, node_to_str)

    if isinstance(node, Histogram):
        return histogram_to_str(node, feature_names)

    if isinstance(node, Product):
        children_strs = map(lambda child: spn_to_str_equation(child, feature_names, node_to_str), node.children)
        return "(" + " * ".join(children_strs) + ")"

    if isinstance(node, Sum):
        fmt_chld = lambda w, c: str(w) + "*(" + spn_to_str_equation(c, feature_names, node_to_str) + ")"

        children_strs = map(lambda i: fmt_chld(node.weights[i], node.children[i]), range(len(node.children)))

        return "(" + " + ".join(children_strs) + ")"

    raise Exception('Node type not registered: ' + str(type(node)))


def histogram_tree_to_spn(tree, features):
    node = Histogram(list(map(float, tree.children[1].children)), list(map(float, tree.children[2].children)))

    feature = str(tree.children[0])

    node.scope.append(features.index(feature))

    return node


_str_to_spn = {"histogram": (histogram_tree_to_spn, """
%import common.WORD -> WORDHIST
%import common.DIGIT -> DIGITHIST
HISTVARNAMECHAR: "a".."z"|"A".."Z"|DIGITHIST
HISTVARNAME: HISTVARNAMECHAR+
listhist : "[" [DECIMAL ("," DECIMAL)*] "]"
histogram: "Histogram(" HISTVARNAME "|" listhist ";" listhist ")" 
""")}


def str_to_spn(text, features, str_to_spn_lambdas=_str_to_spn):
    from lark import Lark

    ext_name = "\n".join(map(lambda s: "    | " + s + " \n", str_to_spn_lambdas.keys()))

    ext_grammar = "\n".join([s for _, s in str_to_spn_lambdas.values()])

    grammar = r"""
%import common.DECIMAL -> DECIMAL
%import common.WS
%ignore WS

?node: prodnode
    | sumnode
""" + ext_name + r"""

prodnode: "(" [node ("*" node)*] ")"
sumnode: "(" [DECIMAL "*" node ("+" DECIMAL "*" node)*] ")"

                """ + ext_grammar

    parser = Lark(grammar, start='node')

    tree = parser.parse(text)

    def tree_to_spn(tree, features=[]):
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
            node = Product()
            for c in tree.children:
                node.children.append(tree_to_spn(c, features))
            return node

        if tnode in str_to_spn_lambdas:
            return str_to_spn_lambdas[tnode][0](tree, features)

        raise Exception('Node type not registered: ' + tnode)

    spn = tree_to_spn(tree, features)

    rebuild_scopes_bottom_up(spn)
    assert is_valid(spn)
    spn = prune(spn)
    assert is_valid(spn)
    assign_ids(spn)
    return spn
