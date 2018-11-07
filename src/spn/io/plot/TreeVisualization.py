import ete3
import matplotlib

from spn.structure.Base import Node, Product, Sum, Leaf, Context


def spn_to_ete(spn, context=None, unroll=False):
    tree = ete3.Tree()
    tree.name = spn.name

    queue = []

    if not isinstance(spn, Leaf):
        for child in spn.children:
            if unroll:
                if child in queue:
                    return '-> ' + spn.id
                else:
                    queue.append(child)
            tree.add_child(spn_to_ete(child, context=context, unroll=unroll))
    elif context is not None:
        feature_names = ', '.join([context.feature_names[i] for i in spn.scope])
        tree.name += ': ' + feature_names 

    return tree


def get_newick(spn, context=None, unroll_dag=False):
    tree = spn_to_ete(spn, context, unroll_dag)
    return tree.write(format=1)


def plot_spn(spn, context=None, unroll=False, file_name=None):
    tree = spn_to_ete(spn, context, unroll)

    if file_name is not None:
        tree.render(file_name)
