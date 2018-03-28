'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Pruning import prune
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, Leaf, get_nodes_by_type, rebuild_scopes_bottom_up, assign_ids


def to_JSON(node):
    import json

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return {obj.__class__.__name__: obj.__dict__}

    return json.dumps(node, default=dumper)


def to_str_ref_graph(node, leaf_to_str, feature_names=None):
    node_name = lambda node: node.__class__.__name__ + "Node_" + str(node.id)

    if isinstance(node, Leaf):
        return node_name(node) + " " + leaf_to_str(node, feature_names) + "\n"

    if isinstance(node, Product):
        pd = ", ".join(map(lambda c: node_name(c), node.children))

        chld_str = "".join(map(lambda c: to_str_ref_graph(c, leaf_to_str, feature_names), node.children))
        chld_str = chld_str.replace("\n", "\n\t")

        return "%s ProductNode(%s){\n\t%s}\n" % (node_name(node), pd, chld_str)

    if isinstance(node, Sum):
        w = node.weights
        ch = node.children
        sumw = ", ".join(map(lambda i: "%s*%s" % (w[i], node_name(ch[i])), range(len(ch))))

        child_str = "".join(map(lambda c: to_str_ref_graph(c, leaf_to_str, feature_names), node.children))
        child_str = child_str.replace("\n", "\n\t")

        return "%s SumNode(%s){\n\t%s}\n" % (node_name(node), sumw, child_str)

    raise Exception('Node type not registered: ' + str(type(node)))


def to_str_equation(node, leaf_to_str, feature_names=None):
    if isinstance(node, Leaf):
        return leaf_to_str(node, feature_names)

    if isinstance(node, Product):
        children_strs = map(lambda child: to_str_equation(child, leaf_to_str, feature_names), node.children)
        return "(" + " * ".join(children_strs) + ")"

    if isinstance(node, Sum):
        fmt_chld = lambda w, c: str(w) + "*(" + to_str_equation(c, leaf_to_str, feature_names) + ")"

        children_strs = map(lambda i: fmt_chld(node.weights[i], node.children[i]), range(len(node.children)))

        return "(" + " + ".join(children_strs) + ")"

    raise Exception('Node type not registered: ' + str(type(node)))


def str_to_spn(text, features, str_to_spn_lambdas={}):
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
