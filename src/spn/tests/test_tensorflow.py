import unittest

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric, learn_mspn
from spn.gpu.TensorFlow import spn_to_tf_graph, eval_tf, likelihood_loss, tf_graph_to_spn
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
import numpy as np


from spn.structure.leaves.parametric.Parametric import Gaussian
import tensorflow as tf


class TestTensorflow(unittest.TestCase):


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

            tf_ll_opt = session.run(tf_graph, feed_dict={data_placeholder: data}).reshape(-1,1)

            tf_graph_to_spn(variable_dict)

        py_ll_opt = log_likelihood(spn, data)

        # print(tf_ll_opt.sum(), py_ll_opt.sum())

        self.assertTrue(np.all(np.isclose(tf_ll_opt, py_ll_opt)))

        self.assertLess(py_ll.sum(), tf_ll_opt.sum())


if __name__ == '__main__':
    unittest.main()
