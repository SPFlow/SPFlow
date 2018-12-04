from ete3 import Tree, TreeStyle, faces, AttrFace, TextFace, NodeStyle
from spn.io.Text import spn_to_str_equation

from spn.structure.Base import Sum, Leaf, Product

_symbols = {Sum: "Σ", Product: "Π"}


def set_symbol(node_type, symbol):
    _symbols[node_type] = symbol


def spn_to_ete(spn, context=None, unroll=False, symbols=_symbols):
    assert spn is not None

    tree = Tree()
    tree.id = spn.id
    tree.node_type = type(spn)
    tree.name = symbols.get(tree.node_type, spn.name)

    queue = []

    if not isinstance(spn, Leaf):
        for i, child in enumerate(spn.children):
            if unroll:
                if child in queue:
                    return "-> " + spn.id
                else:
                    queue.append(child)
            c = spn_to_ete(child, context=context, unroll=unroll)
            if isinstance(spn, Sum):
                c.support = spn.weights[i]
            tree.add_child(c)
    else:
        feature_names = None
        if context is not None:
            feature_names = context.feature_names

        try:
            tree.name = spn_to_str_equation(spn, feature_names=feature_names)
        except:
            if feature_names is None:
                feature_names = []
            tree.name += "(%s)" % ",".join(feature_names)

    return tree


def get_newick(spn, context=None, unroll_dag=False):
    tree = spn_to_ete(spn, context, unroll_dag)
    return tree.write(format=1)


def plot_spn(spn, context=None, unroll=False, file_name=None, show_ids=False):
    assert spn is not None

    lin_style = TreeStyle()

    def my_layout(node):

        style = NodeStyle()
        style["size"] = 0
        style["vt_line_color"] = "#A0A0A0"
        style["hz_line_color"] = "#A0A0A0"
        style["vt_line_type"] = 0  # 0 solid, 1 dashed, 2 dotted
        style["hz_line_type"] = 0
        node.set_style(style)

        if node.is_leaf():
            name_face = AttrFace("name", fsize=8, ftype="Times")
        else:
            name_face = TextFace(node.name, fsize=18, ftype="Times")
            if node.node_type == Sum:
                for child in node.children:
                    label = TextFace(round(child.support, 3), fsize=6)
                    child.add_face(label, column=1, position="branch-bottom")
        if show_ids:
            node.add_face(AttrFace("id", fsize=6), column=1, position="branch-top")
        faces.add_face_to_node(name_face, node, column=1, position="branch-right")

    lin_style.layout_fn = my_layout
    lin_style.show_leaf_name = False
    lin_style.show_scale = False

    tree = spn_to_ete(spn, context, unroll)

    if file_name is not None:
        return tree.render(file_name, tree_style=lin_style)
