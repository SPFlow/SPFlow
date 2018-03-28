'''
Created on March 27, 2018

@author: Alejandro Molina
'''
from joblib import Memory

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.StructureLearning import learn_structure
from spn.algorithms.splitting.KMeans import split_rows_KMeans
from spn.algorithms.splitting.RDC import split_cols_RDC
from spn.io.Text import str_to_spn, to_str_ref_graph
from spn.leaves.Histograms import Histogram, histogram_likelihood, Histogram_str_to_spn, add_domains, \
    create_histogram_leaf, histogram_to_str
from spn.structure.Base import Product, Sum, Leaf, Context, assign_ids, get_number_of_edges
import tensorflow as tf
import numpy as np


def get_tf_graph(node, data, log_space=True):
    # data is a placeholder, with shape same as numpy data

    if not isinstance(node, Leaf):
        childrenprob = [get_tf_graph(c, data, log_space) for c in node.children]

    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        if isinstance(node, Product):
            if log_space:
                return tf.add_n(childrenprob)
            else:
                return tf.reduce_prod(childrenprob, axis=0)

        if isinstance(node, Sum):
            # TODO: make weights as variables
            if log_space:
                w_childrenprob = tf.stack([np.log(node.weights[i]) + ctf for i, ctf in enumerate(childrenprob)], axis=1)
                return tf.reduce_logsumexp(w_childrenprob, axis=1)
            else:
                w_childrenprob = tf.stack([node.weights[i] * ctf for i, ctf in enumerate(childrenprob)], axis=1)
                return tf.reduce_sum(w_childrenprob, axis=1)

        if isinstance(node, Histogram):
            densities = tf.constant(node.densities)

            inps = np.arange(int(max(node.breaks))).reshape((-1, 1))

            hll = histogram_likelihood(node, inps)
            if not log_space:
                hll = np.exp(hll)

            lls = tf.constant(hll)

            col = data[:, node.scope[0]]

            return tf.gather(lls, col)

memory = Memory(cachedir="cache", verbose=0, compress=9)


@memory.cache
def learn(data, ds_context):
    spn = learn_structure(data, ds_context, split_rows_KMeans, split_cols_RDC, create_histogram_leaf)

    return spn


if __name__ == '__main__':
    with open('test_data.txt', 'r') as myfile:
        words = myfile.readline().strip()
        words = words[2:]
        words = np.array(words.split(';'))

    data = np.loadtxt("test_data.txt", delimiter=";")

    top_features = 2
    # data = data[:,0:top_features]
    # words = words[0:top_features]

    F = len(words)

    ds_context = Context()
    ds_context.statistical_type = np.asarray(["discrete"] * F)
    ds_context.distribution_family = np.asarray(["poisson"] * F)
    add_domains(data, ds_context)

    spn = learn(data, ds_context)

    print(to_str_ref_graph(spn, histogram_to_str))
    print(get_number_of_edges(spn))

    data_placeholder = tf.placeholder(tf.int32, data.shape)

    tf_graph = get_tf_graph(spn, data_placeholder, False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outll = sess.run(tf_graph, feed_dict={data_placeholder: data})

        #print(outll)

        py_ll = log_likelihood(spn, data, histogram_likelihood)

        print(np.all(np.isclose(py_ll - np.log(outll), 0)))

        tf.summary.FileWriter('tfgraph', sess.graph)
