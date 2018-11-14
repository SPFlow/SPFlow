from ete3 import Tree, TreeStyle, faces, AttrFace, TextFace

import matplotlib

from spn.structure.Base import Node, Product, Sum, Leaf, Context



def spn_to_ete(spn, context=None, unroll=False):
    tree = Tree()
    tree.name = spn.name

    queue = []

    if not isinstance(spn, Leaf):
        for i, child in enumerate(spn.children):
            if unroll:
                if child in queue:
                    return '-> ' + spn.id
                else:
                    queue.append(child)
            c = spn_to_ete(child, context=context, unroll=unroll)
            if isinstance(spn, Sum):
                c.support = spn.weights[i]
            tree.add_child(c)
    elif context is not None:
        feature_names = ', '.join([context.feature_names[i] for i in spn.scope])
        tree.name += ': ' + feature_names 

    return tree


def get_newick(spn, context=None, unroll_dag=False):
    tree = spn_to_ete(spn, context, unroll_dag)
    return tree.write(format=1)


def plot_spn(spn, context=None, unroll=False, file_name=None):

    lin_style = TreeStyle()
    def my_layout(node):
        if node.is_leaf():
            name_face = AttrFace("name")
        else:
            name_face = TextFace(node.name[:3], fsize=10)
            if node.name[:3] == 'Sum':
                for child in node.children:
                    label = TextFace(round(child.support,3))
                    child.add_face(label, column=1, position="branch-bottom")
        faces.add_face_to_node(name_face, node, column=1, position="branch-right")
        
    lin_style.layout_fn = my_layout
    lin_style.show_leaf_name = False

    tree = spn_to_ete(spn, context, unroll)

    if file_name is not None:
        return tree.render(file_name, tree_style=lin_style)
