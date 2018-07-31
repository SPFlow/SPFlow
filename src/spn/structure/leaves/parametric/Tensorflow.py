'''
Created on March 21, 2018

@author: Alejandro Molina
'''


import tensorflow as tf

from spn.gpu.TensorFlow import add_node_to_tf_graph, add_tf_graph_to_node
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
import numpy as np

def gaussian_to_tf_graph(node, data_placeholder, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        mean = tf.get_variable("mean", initializer=node.mean, dtype=dtype)
        stdev = tf.get_variable("stdev", initializer=node.stdev, dtype=dtype)
        variable_dict[node] = (mean, stdev)
        stdev = tf.maximum(stdev, 0.001)
        if log_space:
            return tf.distributions.Normal(loc=mean, scale=stdev).log_prob(data_placeholder[:, node.scope[0]])

        return tf.distributions.Normal(loc=mean, scale=stdev).prob(data_placeholder[:, node.scope[0]])


def categorical_to_tf_graph(node, data_placeholder, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        p = np.array(node.p, dtype=dtype)
        softmaxInverse = np.log(p / np.max(p)).astype(dtype)
        probs = tf.nn.softmax(tf.get_variable("p", initializer=tf.constant(softmaxInverse)))
        variable_dict[node] = (probs)
        if log_space:
            return tf.distributions.Categorical(probs=probs).log_prob(data_placeholder[:, node.scope[0]])

        return tf.distributions.Categorical(probs=probs).prob(data_placeholder[:, node.scope[0]])


def tf_graph_to_gaussian(node, tfvar):
    node.mean = tfvar[0].eval()
    node.stdev = tfvar[1].eval()


def tf_graph_to_categorical(node, tfvar):
    node.p = tfvar.eval().tolist()

def add_parametric_tensorflow_support():
    add_node_to_tf_graph(Gaussian, gaussian_to_tf_graph)
    add_node_to_tf_graph(Categorical, categorical_to_tf_graph)
    add_tf_graph_to_node(Gaussian, tf_graph_to_gaussian)
    add_tf_graph_to_node(Categorical, tf_graph_to_categorical)
