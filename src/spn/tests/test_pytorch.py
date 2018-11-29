import unittest

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric, learn_mspn
# from spn.gpu.PyTorch import spn_to_pytorch_graph, eval_pytorch, likelihood_loss, pytorch_graph_to_spn
from spn.structure.StatisticalTypes import MetaType
import numpy as np

from spn.structure.leaves.parametric.Parametric import Gaussian
import torch


class TestPytorch(unittest.Testcase):

    def test_eval_gaussian(self):
        np.random.seed(17)
        data = np.random.normal(10, 0.01, size=2000).tolist() + \
            np.random.normal(30, 10, size=2000).tolist()
        data = np.array(data).reshape((-1, 10))
        data = data.astype(np.float32)

        ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1],
                             parametric_types=[Gaussian] * data.shape[1])
        spn = learn_parametric(data, ds_context)

        ll = log_likelihood(spn, data)

        tf_ll = eval_pytorch(spn, data)

        self.assertTrue(np.all(np.isclose(ll, tf_ll)))


if __name__ == '__main__':
    unittest.main()
