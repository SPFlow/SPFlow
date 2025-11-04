import unittest

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from spn.algorithms.TransformStructure import Copy
from spn.io.Text import spn_to_str_equation

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric, learn_mspn, learn_mspn_with_missing
from spn.gpu.TensorFlow import spn_to_tf_graph, eval_tf, likelihood_loss, tf_graph_to_spn
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
import numpy as np

from spn.structure.leaves.parametric.Parametric import (
    Gaussian,
    Exponential,
    Gamma,
    LogNormal,
    Poisson,
    Bernoulli,
    Categorical,
)


class TestTensorflow(unittest.TestCase):
    def test_eval_parametric(self):
        data = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32).reshape((1, 7))

        spn = (
            Gaussian(mean=1.0, stdev=1.0, scope=[0])
            * Exponential(l=1.0, scope=[1])
            * Gamma(alpha=1.0, beta=1.0, scope=[2])
            * LogNormal(mean=1.0, stdev=1.0, scope=[3])
            * Poisson(mean=1.0, scope=[4])
            * Bernoulli(p=0.6, scope=[5])
            * Categorical(p=[0.1, 0.2, 0.7], scope=[6])
        )

        ll = log_likelihood(spn, data)

        tf_ll = eval_tf(spn, data)

        self.assertTrue(np.all(np.isclose(ll, tf_ll)))

        spn_copy = Copy(spn)

        tf_graph, data_placeholder, variable_dict = spn_to_tf_graph(spn_copy, data, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_graph_to_spn(variable_dict)

        str_val = spn_to_str_equation(spn)
        str_val2 = spn_to_str_equation(spn_copy)

        self.assertEqual(str_val, str_val2)

    def test_eval_gaussian(self):
        np.random.seed(17)
        data = np.random.normal(10, 0.01, size=2000).tolist() + np.random.normal(30, 10, size=2000).tolist()
        data = np.array(data).reshape((-1, 10))
        data = data.astype(np.float32)

        ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1], parametric_types=[Gaussian] * data.shape[1])

        spn = learn_parametric(data, ds_context)

        ll = log_likelihood(spn, data)

        tf_ll = eval_tf(spn, data)

        self.assertTrue(np.all(np.isclose(ll, tf_ll)))

    def test_eval_histogram(self):
        np.random.seed(17)
        data = np.random.normal(10, 0.01, size=2000).tolist() + np.random.normal(30, 10, size=2000).tolist()
        data = np.array(data).reshape((-1, 10))
        data[data < 0] = 0
        data = data.astype(int)

        ds_context = Context(meta_types=[MetaType.DISCRETE] * data.shape[1])
        ds_context.add_domains(data)

        spn = learn_mspn(data, ds_context)
        learn_mspn_with_missing(data, ds_context)

        ll = log_likelihood(spn, data)

        tf_ll = eval_tf(spn, data)

        self.assertTrue(np.all(np.isclose(ll, tf_ll)))

    def test_optimization(self):
        np.random.seed(17)
        data = np.random.normal(10, 0.01, size=2000).tolist() + np.random.normal(30, 10, size=2000).tolist()
        data = np.array(data).reshape((-1, 10))
        data = data.astype(np.float32)

        ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1], parametric_types=[Gaussian] * data.shape[1])

        spn = learn_parametric(data, ds_context)

        spn.weights = [0.8, 0.2]

        py_ll = log_likelihood(spn, data)

        tf_graph, data_placeholder, variable_dict = spn_to_tf_graph(spn, data)

        loss = likelihood_loss(tf_graph)

        output = tf.train.AdamOptimizer(0.001).minimize(loss)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for step in range(50):
                session.run(output, feed_dict={data_placeholder: data})
                # print("loss:", step, session.run(-loss, feed_dict={data_placeholder: data}))

            tf_ll_opt = session.run(tf_graph, feed_dict={data_placeholder: data}).reshape(-1, 1)

            tf_graph_to_spn(variable_dict)

        py_ll_opt = log_likelihood(spn, data)

        # print(tf_ll_opt.sum(), py_ll_opt.sum())

        self.assertTrue(np.all(np.isclose(tf_ll_opt, py_ll_opt)))

        self.assertLess(py_ll.sum(), tf_ll_opt.sum())


if __name__ == "__main__":
    unittest.main()
