"""
Created on March 27, 2018

@author: Alejandro Molina
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from typing import Union, Tuple, List

from tensorflow.python.client import timeline

from spn.algorithms.TransformStructure import Copy
from spn.structure.Base import Product, Sum, eval_spn_bottom_up, Node
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.histogram.Inference import histogram_log_likelihood
from spn.structure.leaves.parametric.Parametric import Gaussian
import logging

logger = logging.getLogger(__name__)


def log_sum_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
    assert log_space
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        softmaxInverse = np.log(node.weights / np.max(node.weights)).astype(dtype)
        tfweights = tf.nn.softmax(tf.get_variable("weights", initializer=tf.constant(softmaxInverse)))
        variable_dict[node] = tfweights
        childrenprob = tf.stack(children, axis=1)
        return tf.reduce_logsumexp(childrenprob + tf.log(tfweights), axis=1)


def tf_graph_to_sum(node, tfvar):
    node.weights = tfvar.tolist()


def log_prod_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
    assert log_space
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        return tf.add_n(children)


def histogram_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        inps = np.arange(int(max(node.breaks))).reshape((-1, 1))
        tmpscope = node.scope[0]
        node.scope[0] = 0
        hll = histogram_log_likelihood(node, inps)
        node.scope[0] = tmpscope
        if not log_space:
            hll = np.exp(hll)

        lls = tf.constant(hll.astype(dtype))

        col = data_placeholder[:, node.scope[0]]

        return tf.squeeze(tf.gather(lls, col))


_node_log_tf_graph = {Sum: log_sum_to_tf_graph, Product: log_prod_to_tf_graph, Histogram: histogram_to_tf_graph}


def add_node_to_tf_graph(node_type, lambda_func):
    _node_log_tf_graph[node_type] = lambda_func


_tf_graph_to_node = {Sum: tf_graph_to_sum}


def add_tf_graph_to_node(node_type, lambda_func):
    _tf_graph_to_node[node_type] = lambda_func


def spn_to_tf_graph(node, data, batch_size=None, node_tf_graph=_node_log_tf_graph, log_space=True, dtype=None):
    tf.reset_default_graph()
    if not dtype:
        dtype = data.dtype
    # data is a placeholder, with shape same as numpy data
    data_placeholder = tf.placeholder(data.dtype, (batch_size, data.shape[1]))
    variable_dict = {}
    tf_graph = eval_spn_bottom_up(
        node,
        node_tf_graph,
        data_placeholder=data_placeholder,
        log_space=log_space,
        variable_dict=variable_dict,
        dtype=dtype,
    )
    return tf_graph, data_placeholder, variable_dict


def tf_graph_to_spn(variable_dict, tf_graph_to_node=_tf_graph_to_node):
    tensors = []

    for n, tfvars in variable_dict.items():
        tensors.append(tfvars)

    variable_list = tf.get_default_session().run(tensors)

    for i, (n, tfvars) in enumerate(variable_dict.items()):
        tf_graph_to_node[type(n)](n, variable_list[i])


def likelihood_loss(tf_graph):
    # minimize negative log likelihood
    return -tf.reduce_sum(tf_graph)


def optimize_tf(
    spn: Node,
    data: np.ndarray,
    epochs=1000,
    batch_size: int = None,
    optimizer: tf.train.Optimizer = None,
    return_loss=False,
) -> Union[Tuple[Node, List[float]], Node]:
    """
    Optimize weights of an SPN with a tensorflow stochastic gradient descent optimizer, maximizing the likelihood
    function.
    :param spn: SPN which is to be optimized
    :param data: Input data
    :param epochs: Number of epochs
    :param batch_size: Size of each minibatch for SGD
    :param optimizer: Optimizer procedure
    :param return_loss: Whether to also return the list of losses for each epoch or not
    :return: If `return_loss` is true, a copy of the optimized SPN and the list of the losses for each epoch is
    returned, else only a copy of the optimized SPN is returned
    """
    # Make sure, that the passed SPN is not modified
    spn_copy = Copy(spn)

    # Compile the SPN to a static tensorflow graph
    tf_graph, data_placeholder, variable_dict = spn_to_tf_graph(spn_copy, data, batch_size)

    # Optimize the tensorflow graph
    loss_list = optimize_tf_graph(
        tf_graph, variable_dict, data_placeholder, data, epochs=epochs, batch_size=batch_size, optimizer=optimizer
    )

    # Return loss as well if flag is set
    if return_loss:
        return spn_copy, loss_list

    return spn_copy


def optimize_tf_graph(
    tf_graph, variable_dict, data_placeholder, data, epochs=1000, batch_size=None, optimizer=None
) -> List[float]:
    if optimizer is None:
        optimizer = tf.train.GradientDescentOptimizer(0.001)
    loss = -tf.reduce_sum(tf_graph)
    opt_op = optimizer.minimize(loss)

    # Collect loss
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if not batch_size:
            batch_size = data.shape[0]
        batches_per_epoch = data.shape[0] // batch_size

        # Iterate over epochs
        for i in range(epochs):

            # Collect loss over batches for one epoch
            epoch_loss = 0.0

            # Iterate over batches
            for j in range(batches_per_epoch):
                data_batch = data[j * batch_size : (j + 1) * batch_size, :]
                _, batch_loss = sess.run([opt_op, loss], feed_dict={data_placeholder: data_batch})
                epoch_loss += batch_loss

            # Build mean
            epoch_loss /= data.shape[0]

            logger.debug("Epoch: %s, Loss: %s", i, epoch_loss)
            loss_list.append(epoch_loss)

        tf_graph_to_spn(variable_dict)

    return loss_list


def eval_tf(spn, data, save_graph_path=None, dtype=np.float32):
    tf_graph, placeholder, _ = spn_to_tf_graph(spn, data, dtype=dtype)
    return eval_tf_graph(tf_graph, placeholder, data, save_graph_path=save_graph_path)


def eval_tf_graph(tf_graph, data_placeholder, data, save_graph_path=None):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tf_graph, feed_dict={data_placeholder: data})

        if save_graph_path is not None:
            tf.summary.FileWriter(save_graph_path, sess.graph)

        return result.reshape(-1, 1)


def eval_tf_trace(spn, data, log_space=True, save_graph_path=None):
    data_placeholder = tf.placeholder(data.dtype, data.shape)
    import time

    tf_graph = spn_to_tf_graph(spn, data_placeholder, log_space)
    run_metadata = None
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(tf.global_variables_initializer())

        start = time.perf_counter()
        result = sess.run(tf_graph, feed_dict={data_placeholder: data}, options=run_options, run_metadata=run_metadata)
        end = time.perf_counter()

        e2 = end - start

        logger.info(e2)

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()

        import json

        traceEvents = json.loads(ctf)["traceEvents"]
        elapsed = max([o["ts"] + o["dur"] for o in traceEvents if "ts" in o and "dur" in o]) - min(
            [o["ts"] for o in traceEvents if "ts" in o]
        )
        return result, elapsed

        if save_graph_path is not None:
            summary_fw = tf.summary.FileWriter(save_graph_path, sess.graph)
            if trace:
                summary_fw.add_run_metadata(run_metadata, "run")

        return result, -1



def add_parametric_tensorflow_support():
    from spn.structure.leaves.parametric.Parametric import (
        Gaussian,
        Categorical,
        LogNormal,
        Exponential,
        Gamma,
        Poisson,
        Bernoulli,
    )
    from spn.structure.leaves.parametric.Tensorflow import (
        gaussian_to_tf_graph,
        exponential_to_tf_graph,
        gamma_to_tf_graph,
        lognormal_to_tf_graph,
        poisson_to_tf_graph,
        bernoulli_to_tf_graph,
        categorical_to_tf_graph,

        tf_graph_to_gaussian,
        tf_graph_to_gamma,
        tf_graph_to_exponential,
        tf_graph_to_poisson,
        tf_graph_to_bernoulli,
        tf_graph_to_categorical
    )

    add_node_to_tf_graph(Gaussian, gaussian_to_tf_graph)
    add_node_to_tf_graph(Exponential, exponential_to_tf_graph)
    add_node_to_tf_graph(Gamma, gamma_to_tf_graph)
    add_node_to_tf_graph(LogNormal, lognormal_to_tf_graph)
    add_node_to_tf_graph(Poisson, poisson_to_tf_graph)
    add_node_to_tf_graph(Bernoulli, bernoulli_to_tf_graph)
    add_node_to_tf_graph(Categorical, categorical_to_tf_graph)

    add_tf_graph_to_node(Gaussian, tf_graph_to_gaussian)
    add_tf_graph_to_node(Exponential, tf_graph_to_exponential)
    add_tf_graph_to_node(Gamma, tf_graph_to_gamma)
    add_tf_graph_to_node(LogNormal, tf_graph_to_gaussian)
    add_tf_graph_to_node(Poisson, tf_graph_to_poisson)
    add_tf_graph_to_node(Bernoulli, tf_graph_to_bernoulli)
    add_tf_graph_to_node(Categorical, tf_graph_to_categorical)

add_parametric_tensorflow_support()