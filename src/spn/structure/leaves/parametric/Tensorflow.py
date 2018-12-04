"""
Created on March 21, 2018

@author: Alejandro Molina
"""

import tensorflow as tf

from spn.gpu.TensorFlow import add_node_to_tf_graph, add_tf_graph_to_node
from spn.structure.leaves.parametric.Parametric import (
    Gaussian,
    Categorical,
    LogNormal,
    Exponential,
    Gamma,
    Poisson,
    Bernoulli,
)
import numpy as np


def gaussian_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        mean = tf.get_variable("mean", initializer=node.mean, dtype=dtype)
        stdev = tf.get_variable("stdev", initializer=node.stdev, dtype=dtype)
        variable_dict[node] = (mean, stdev)
        stdev = tf.maximum(stdev, 0.001)
        dist = tf.distributions.Normal(loc=mean, scale=stdev)
        if log_space:
            return dist.log_prob(data_placeholder[:, node.scope[0]])

        return dist.prob(data_placeholder[:, node.scope[0]])


def exponential_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        l = tf.get_variable("rate", initializer=node.l, dtype=dtype)
        variable_dict[node] = l
        l = tf.maximum(l, 0.001)
        dist = tf.distributions.Exponential(l=l)
        if log_space:
            return dist.log_prob(data_placeholder[:, node.scope[0]])

        return dist.prob(data_placeholder[:, node.scope[0]])


def poisson_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        mean = tf.maximum(tf.get_variable("lambda", initializer=node.mean, dtype=dtype), 0.001)
        variable_dict[node] = mean
        dist = tf.distributions.Poisson(rate=mean)
        if log_space:
            return dist.log_prob(data_placeholder[:, node.scope[0]])

        return dist.prob(data_placeholder[:, node.scope[0]])


def bernoulli_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        p = tf.maximum(tf.get_variable("p", initializer=node.p, dtype=dtype), 0.001)
        variable_dict[node] = p
        dist = tf.distributions.Bernoulli(probs=p)
        if log_space:
            return dist.log_prob(data_placeholder[:, node.scope[0]])

        return dist.prob(data_placeholder[:, node.scope[0]])


def gamma_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        alpha = tf.maximum(tf.get_variable("alpha", initializer=node.alpha, dtype=dtype), 0.001)
        beta = tf.maximum(tf.get_variable("beta", initializer=node.beta, dtype=dtype), 0.001)
        variable_dict[node] = (alpha, beta)
        dist = tf.distributions.Gamma(concentration=alpha, rate=beta)
        if log_space:
            return dist.log_prob(data_placeholder[:, node.scope[0]])

        return dist.prob(data_placeholder[:, node.scope[0]])


def lognormal_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        mean = tf.get_variable("mean", initializer=node.mean, dtype=dtype)
        stdev = tf.get_variable("stdev", initializer=node.stdev, dtype=dtype)
        variable_dict[node] = (mean, stdev)
        stdev = tf.maximum(stdev, 0.001)
        dist = tf.distributions.TransformedDistribution(
            distribution=tf.distributions.Normal(loc=mean, scale=stdev),
            bijector=tf.distributions.bijectors.Exp(),
            name="LogNormalDistribution",
        )
        if log_space:
            return dist.log_prob(data_placeholder[:, node.scope[0]])

        return dist.prob(data_placeholder[:, node.scope[0]])


def categorical_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        p = np.array(node.p, dtype=dtype)
        softmaxInverse = np.log(p / np.max(p)).astype(dtype)
        probs = tf.nn.softmax(tf.get_variable("p", initializer=tf.constant(softmaxInverse)))
        variable_dict[node] = probs
        if log_space:
            return tf.distributions.Categorical(probs=probs).log_prob(data_placeholder[:, node.scope[0]])

        return tf.distributions.Categorical(probs=probs).prob(data_placeholder[:, node.scope[0]])


def tf_graph_to_gaussian(node, tfvar):
    node.mean = tfvar[0].eval()
    node.stdev = tfvar[1].eval()


def tf_graph_to_gamma(node, tfvar):
    node.alpha = tfvar[0].eval()
    node.beta = tfvar[1].eval()


def tf_graph_to_categorical(node, tfvar):
    node.p = tfvar.eval().tolist()


def tf_graph_to_exponential(node, tfvar):
    node.l = tfvar.eval()


def tf_graph_to_poisson(node, tfvar):
    node.mean = tfvar.eval()


def tf_graph_to_bernoulli(node, tfvar):
    node.p = tfvar.eval()


def add_parametric_tensorflow_support():
    add_node_to_tf_graph(Gaussian, gaussian_to_tf_graph)
    add_node_to_tf_graph(Exponential, exponential_to_tf_graph)
    add_node_to_tf_graph(Gamma, gamma_to_tf_graph)
    add_node_to_tf_graph(LogNormal, lognormal_to_tf_graph)
    add_node_to_tf_graph(Poisson, poisson_to_tf_graph)
    add_node_to_tf_graph(Bernoulli, poisson_to_tf_graph)
    add_node_to_tf_graph(Categorical, categorical_to_tf_graph)

    add_tf_graph_to_node(Gaussian, tf_graph_to_gaussian)
    add_tf_graph_to_node(Exponential, tf_graph_to_exponential)
    add_tf_graph_to_node(Gamma, tf_graph_to_gamma)
    add_tf_graph_to_node(LogNormal, tf_graph_to_gaussian)
    add_tf_graph_to_node(Poisson, tf_graph_to_poisson)
    add_tf_graph_to_node(Bernoulli, tf_graph_to_bernoulli)
    add_tf_graph_to_node(Categorical, tf_graph_to_categorical)
