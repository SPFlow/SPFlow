import unittest
from spflow.python.inference.nodes import likelihood, log_likelihood
from spflow.python.structure.nodes.leaves.parametric.parametric import Gaussian
import numpy as np
from spflow.python.structure.nodes import ISumNode, IProductNode
from spflow.python.structure.network_type import SPN


class TestInference(unittest.TestCase):
    def test_inference_1(self):
        spn = ISumNode(
            children=[
                IProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        ISumNode(
                            children=[
                                IProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                                IProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                            ],
                            scope=[1, 2],
                            weights=np.array([0.3, 0.7]),
                        ),
                    ],
                    scope=[0, 1, 2],
                ),
                IProductNode(
                    children=[
                        IProductNode(
                            children=[
                                Gaussian(scope=[0], mean=0, stdev=1.0),
                                Gaussian(scope=[1], mean=0, stdev=1.0),
                            ],
                            scope=[0, 1],
                        ),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[0, 1, 2],
                ),
            ],
            scope=[0, 1, 2],
            weights=np.array([0.4, 0.6]),
        )

        result = likelihood(SPN(), spn, np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
        self.assertAlmostEqual(result[0][0], 0.023358)

    def test_inference_2(self):
        spn = ISumNode(
            children=[
                IProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        ISumNode(
                            children=[
                                IProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                                IProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                            ],
                            scope=[1, 2],
                            weights=np.array([0.3, 0.7]),
                        ),
                    ],
                    scope=[0, 1, 2],
                ),
                IProductNode(
                    children=[
                        IProductNode(
                            children=[
                                Gaussian(scope=[0], mean=0, stdev=1.0),
                                Gaussian(scope=[1], mean=0, stdev=1.0),
                            ],
                            scope=[0, 1],
                        ),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[0, 1, 2],
                ),
            ],
            scope=[0, 1, 2],
            weights=np.array([0.4, 0.6]),
        )

        result = log_likelihood(SPN(), spn, np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
        self.assertAlmostEqual(result[0][0], -3.7568156)

    def test_inference_3(self):
        spn = ISumNode(
            children=[
                IProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        ISumNode(
                            children=[
                                IProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                                IProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                            ],
                            scope=[1, 2],
                            weights=np.array([0.3, 0.7]),
                        ),
                    ],
                    scope=[0, 1, 2],
                ),
                IProductNode(
                    children=[
                        IProductNode(
                            children=[
                                Gaussian(scope=[0], mean=0, stdev=1.0),
                                Gaussian(scope=[1], mean=0, stdev=1.0),
                            ],
                            scope=[0, 1],
                        ),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[0, 1, 2],
                ),
            ],
            scope=[0, 1, 2],
            weights=np.array([0.4, 0.6]),
        )

        result = likelihood(SPN(), spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
        self.assertAlmostEqual(result[0][0], 0.09653235)

    def test_inference_4(self):
        spn = ISumNode(
            children=[
                IProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        ISumNode(
                            children=[
                                IProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                                IProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                            ],
                            scope=[1, 2],
                            weights=np.array([0.3, 0.7]),
                        ),
                    ],
                    scope=[0, 1, 2],
                ),
                IProductNode(
                    children=[
                        IProductNode(
                            children=[
                                Gaussian(scope=[0], mean=0, stdev=1.0),
                                Gaussian(scope=[1], mean=0, stdev=1.0),
                            ],
                            scope=[0, 1],
                        ),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[0, 1, 2],
                ),
            ],
            scope=[0, 1, 2],
            weights=np.array([0.4, 0.6]),
        )

        result = log_likelihood(SPN(), spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
        self.assertAlmostEqual(result[0][0], -2.33787707)


if __name__ == "__main__":
    unittest.main()
