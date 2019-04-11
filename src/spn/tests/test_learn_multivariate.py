import unittest

from numpy.random.mtrand import RandomState
from scipy.stats import multivariate_normal

from spn.structure.Base import Leaf, bfs, get_topological_order, get_topological_order_layers
import numpy as np


class TestBase(unittest.TestCase):
    def test_learn(self):
        from sklearn.datasets import load_iris

        iris = load_iris()
        X = iris.data
        y = iris.target.reshape(-1, 1)

        train_data = np.hstack((X, y))

        from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
        from spn.structure.leaves.parametric.Parametric import Categorical, MultivariateGaussian
        from spn.structure.Base import Context

        spn_classification = learn_parametric(
            train_data,
            Context(
                parametric_types=[
                    MultivariateGaussian,
                    MultivariateGaussian,
                    MultivariateGaussian,
                    MultivariateGaussian,
                    Categorical,
                ]
            ).add_domains(train_data),
            multivariate_leaf=True,
        )

        # from spn.io.Graphics import plot_spn

        # plot_spn(spn_classification, "basicspn.png")


if __name__ == "__main__":
    unittest.main()
