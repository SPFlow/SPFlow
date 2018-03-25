'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, Leaf


def to_JSON(node):
    import json

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return {obj.__class__.__name__: obj.__dict__}

    return json.dumps(node, default=dumper)


def to_str_equation(node, to_str_equation_lambdas, feature_names=None):
    tnode = type(node)
    if tnode in to_str_equation_lambdas:
        return to_str_equation_lambdas[tnode](node, feature_names)

    if isinstance(node, Product):
        return "(" + " * ".join(
            map(lambda child: to_str_equation(child, to_str_equation_lambdas, feature_names), node.children)) + ")"

    if isinstance(node, Sum):
        sumeq = " + ".join(
            map(lambda i: str(node.weights[i]) + "*(" + to_str_equation(node.children[i], to_str_equation_lambdas,
                                                                        feature_names) + ")",
                range(len(node.children))))
        return "(" + sumeq + ")"

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

    def rebuild_scopes(node):
        # this function is not safe (updates in place)
        if isinstance(node, Leaf):
            return node.scope

        new_scope = set()
        for c in node.children:
            new_scope.update(rebuild_scopes(c))
        node.scope.extend(new_scope)
        return node.scope

    rebuild_scopes(spn)
    assert is_valid(spn)
    return spn
