'''
Created on March 27, 2018

@author: Alejandro Molina
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from spn.algorithms.Inference import histogram_likelihood
from spn.structure.Base import Product, Sum, Leaf
from spn.structure.leaves.Histograms import Histogram


def spn_to_tf_graph(node, data_placeholder, log_space=True):
    # data is a placeholder, with shape same as numpy data

    if not isinstance(node, Leaf):
        childrenprob = [spn_to_tf_graph(c, data_placeholder, log_space) for c in node.children]

    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        if isinstance(node, Product):
            if log_space:
                return tf.add_n(childrenprob)
            else:
                prod_res = None
                for c in childrenprob:
                    if prod_res is None:
                        prod_res = c
                    else:
                        prod_res = tf.multiply(prod_res, c)
                return prod_res

        if isinstance(node, Sum):
            # TODO: make weights as variables
            if log_space:
                w_childrenprob = tf.stack([np.log(node.weights[i]) + ctf for i, ctf in enumerate(childrenprob)], axis=1)
                return tf.reduce_logsumexp(w_childrenprob, axis=1)
            else:
                return tf.add_n([node.weights[i] * ctf for i, ctf in enumerate(childrenprob)])

        if isinstance(node, Histogram):
            inps = np.arange(int(max(node.breaks))).reshape((-1, 1))

            hll = histogram_likelihood(node, inps, log_space=log_space)

            lls = tf.constant(hll)

            col = data_placeholder[:, node.scope[0]]

            return tf.gather(lls, col)


def eval_tf(spn, data, log_space=True, save_graph_path=None, trace=False):
    data_placeholder = tf.placeholder(data.dtype, data.shape[0])
    import time
    tf_graph = spn_to_tf_graph(spn, data_placeholder, log_space)
    run_metadata = None
    with tf.Session() as sess:
        if trace:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(tf.global_variables_initializer())

            start = time.perf_counter()
            result = sess.run(tf_graph, feed_dict={data_placeholder: data}, options=run_options,
                              run_metadata=run_metadata)
            end = time.perf_counter()

            e2 = end - start

            print(e2)

            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()

            import json
            traceEvents = json.loads(ctf)["traceEvents"]
            elapsed = max([o["ts"] + o["dur"] for o in traceEvents if "ts" in o and "dur" in o]) - min(
                [o["ts"] for o in traceEvents if "ts" in o])
            return result, elapsed
        else:
            sess.run(tf.global_variables_initializer())
            result = sess.run(tf_graph, feed_dict={data_placeholder: data})

        if save_graph_path is not None:
            summary_fw = tf.summary.FileWriter(save_graph_path, sess.graph)
            if trace:
                summary_fw.add_run_metadata(run_metadata, "run")

        return result, -1
