import ete3
import matplotlib

from spn.structure.Base import Node, Product, Sum, Leaf, Context


def spn_to_ete(spn, context=None, unroll_dag=False):
    tree = ete3.Tree()
    tree.name = spn.name

    queue = []

    if not isinstance(spn, Leaf):
        for child in spn.children:
            if unroll_dag:
                if child in queue:
                    return '-> to node id' + spn.id
                else:
                    queue.append(child)
            tree.add_child(spn_to_ete(child, context=context, unroll_dag=unroll_dag))
    elif context is not None:
        feature_names = ', '.join([context.feature_names[i] for i in spn.scope])
        tree.name += ': ' + feature_names 

    return tree


def get_newick(spn, context=None, unroll_dag=False):
    tree = spn_to_ete(spn, context, unroll_dag)
    return tree.write(format=1)


def spn_visualize(spn, context=None, unroll_dag=False, output_format='png', file_name=None):
    tree = spn_to_ete(spn, context, unroll_dag)

    if file_name is not None:
        tree.render(file_name)

    return tree

