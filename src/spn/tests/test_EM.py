import unittest

from spn.algorithms.EM import EM_optimization
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric, learn_mspn
from spn.gpu.TensorFlow import spn_to_tf_graph, eval_tf, likelihood_loss, tf_graph_to_spn
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
import numpy as np


from spn.structure.leaves.parametric.Parametric import Gaussian
import tensorflow as tf


class TestEM(unittest.TestCase):


    def test_optimization(self):
        np.random.seed(17)
        data = np.random.normal(10, 5, size=2000).tolist() + np.random.normal(30, 10, size=2000).tolist()
        data = np.array(data).reshape((-1, 10))
        data = data.astype(np.float32)

        ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1], parametric_types=[Gaussian] * data.shape[1])

        spn = learn_parametric(data, ds_context)

        spn.weights = [0.8, 0.2]
        spn.children[0].children[0].mean = 3.0

        py_ll = log_likelihood(spn, data)

        print(spn.weights)
        print(spn.children[0].children[0].mean)

        EM_optimization(spn, data)

        print(spn.weights)
        print(spn.children[0].children[0].mean)

        py_ll_opt = log_likelihood(spn, data)



if __name__ == '__main__':
    unittest.main()
