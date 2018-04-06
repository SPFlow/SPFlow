'''
Created on April 5, 2018

@author: Alejandro Molina
'''
import numpy as np

from spn.algorithms.Inference import likelihood
from spn.io.Text import str_to_spn, to_JSON
from spn.structure.Base import Product, Sum, Leaf, get_nodes_by_type


def sample_induced_trees(node, data, rand_gen, node_likelihood=None):
    all_nodes = get_nodes_by_type(node)
    nr_nodes = len(all_nodes)

    sum_nodes = get_nodes_by_type(node, Sum)
    maxc = max([len(s.children) for s in sum_nodes])
    induced_trees_sum_nodes = np.zeros((data.shape[0], len(sum_nodes), maxc))
    induced_trees_leaf_nodes = np.zeros((data.shape[0], len(get_nodes_by_type(node, Leaf))))

    lls = np.zeros((data.shape[0], nr_nodes))
    likelihood(node, data, node_likelihood=node_likelihood, lls_matrix=lls)

    def sample_induced_trees(node, row_ids):
        if isinstance(node, Leaf):
            induced_trees_leaf_nodes[row_ids, node.id] = 1
            return
        if isinstance(node, Product):
            #induced_trees[row_ids, node.id, 0] = 1
            for c in node.children:
                sample_induced_trees(c, row_ids)
            return
        if isinstance(node, Sum):
            w_children_log_probs = np.zeros((len(row_ids), len(n.weights)))
            for i, c in enumerate(node.children):
                w_children_log_probs[:, i] = lls[row_ids, c.id] + np.log(node.weights[i])

            z_gumbels = rand_gen.gumbel(loc=0, scale=1,
                                        size=(w_children_log_probs.shape[1], w_children_log_probs.shape[0]))
            g_children_log_probs = w_children_log_probs + z_gumbels
            rand_child_branches = np.argmax(g_children_log_probs, axis=0)

            for i, c in enumerate(node.children):
                new_row_ids = row_ids[rand_child_branches == i]
                induced_trees_sum_nodes[new_row_ids, node.id, i] = 1
                sample_induced_trees(c, new_row_ids)
            pass

    sample_induced_trees(node, np.arange(data.shape[0]))

    return induced_trees_sum_nodes, induced_trees_leaf_nodes, lls


if __name__ == '__main__':
    n = str_to_spn("""
            (
            Histogram(W1|[ 0., 1., 2.];[0.3, 0.7])
            *
            Histogram(W2|[ 0., 1., 2.];[0.3, 0.7])
            )    
            """, ["W1", "W2"])

    n = str_to_spn("""
            (0.3 * Histogram(W1|[ 0., 1., 2.];[0.2, 0.8])
            +
            0.7 * Histogram(W1|[ 0., 1., 2.];[0.1, 0.9])
            )    
            """, ["W1", "W2"])

    print(to_JSON(n))

    data = np.vstack((np.asarray([1.5, 0.5]), np.asarray([0.5, 0.5])))

    print(data)
    rand_gen = np.random.RandomState(17)
    print(sample_induced_trees(n, data, rand_gen))
