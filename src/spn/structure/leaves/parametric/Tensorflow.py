"""
Created on March 21, 2018

@author: Alejandro Molina
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp

import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        dist = tf.distributions.Exponential(rate=l)
        if log_space:
            return dist.log_prob(data_placeholder[:, node.scope[0]])

        return dist.prob(data_placeholder[:, node.scope[0]])


def poisson_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        mean = tf.maximum(tf.get_variable("lambda", initializer=node.mean, dtype=dtype), 0.001)
        variable_dict[node] = mean
        dist = tfp.distributions.Poisson(rate=mean)
        if log_space:
            return dist.log_prob(data_placeholder[:, node.scope[0]])

        return dist.prob(data_placeholder[:, node.scope[0]])


def bernoulli_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        p = tf.minimum(tf.maximum(tf.get_variable("p", initializer=node.p, dtype=dtype), 0.00000001), 0.9999999)
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
        # Use tfp.distributions.LogNormal directly for compatibility with newer TFP versions
        dist = tfp.distributions.LogNormal(loc=mean, scale=stdev)
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
    # Round to 7 decimal places to avoid float32 precision artifacts while preserving accuracy
    node.mean = round(float(tfvar[0]), 7)
    node.stdev = round(float(tfvar[1]), 7)


def tf_graph_to_gamma(node, tfvar):
    # Round to 7 decimal places to avoid float32 precision artifacts while preserving accuracy
    node.alpha = round(float(tfvar[0]), 7)
    node.beta = round(float(tfvar[1]), 7)


def tf_graph_to_categorical(node, tfvar):
    # Round to 7 decimal places to avoid float32 precision artifacts while preserving accuracy
    node.p = [round(float(x), 7) for x in tfvar]


def tf_graph_to_exponential(node, tfvar):
    # Round to 7 decimal places to avoid float32 precision artifacts while preserving accuracy
    node.l = round(float(tfvar), 7)


def tf_graph_to_poisson(node, tfvar):
    # Round to 7 decimal places to avoid float32 precision artifacts while preserving accuracy
    node.mean = round(float(tfvar), 7)


def tf_graph_to_bernoulli(node, tfvar):
    # Round to 7 decimal places to avoid float32 precision artifacts while preserving accuracy
    node.p = round(float(tfvar), 7)


