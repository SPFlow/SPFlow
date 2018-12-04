import unittest
import tensorflow as tf
import numpy as np
import spn.experiments.RandomSPNs.RAT_SPN as RAT_SPN
import spn.experiments.RandomSPNs.region_graph as region_graph

import spn.algorithms.Inference as Inference


class TestRatSpn(unittest.TestCase):
    def test_inference_results(self):
        np.random.seed(123)
        tf.set_random_seed(123)

        num_dims = 20

        rg = region_graph.RegionGraph(range(num_dims))
        for _ in range(0, 10):
            rg.random_split(2, 3)

        args = RAT_SPN.SpnArgs()
        args.normalized_sums = True
        spn = RAT_SPN.RatSpn(10, region_graph=rg, name="obj-spn", args=args)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        dummy_input = np.random.normal(0.0, 1.2, [10, num_dims])
        input_ph = tf.placeholder(tf.float32, [10, num_dims])
        output_tensor = spn.forward(input_ph)
        tf_output = sess.run(output_tensor, feed_dict={input_ph: dummy_input})

        output_nodes = spn.get_simple_spn(sess)
        simple_output = []
        for node in output_nodes:
            simple_output.append(Inference.likelihood(node, dummy_input))
        simple_output = np.stack(simple_output)
        deviation = simple_output / np.exp(tf_output)
        rel_error = np.abs(deviation - 1.0)
        # print(rel_error)

        self.assertTrue(np.all(rel_error < 1e-2))


if __name__ == "__main__":
    unittest.main()
