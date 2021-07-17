import unittest
from spn.base.nodes.inference import likelihood, log_likelihood
from spn.base.nodes.leaves.parametric.parametric import Gaussian
import numpy as np
from spn.base.nodes.node import (
    SumNode,
    ProductNode,
)

# TODO:
# test all other parametric types


class TestInference(unittest.TestCase):
    def test_inference_1(self):
        spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        SumNode(
                            children=[
                                ProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                                ProductNode(
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
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        Gaussian(scope=[1], mean=0, stdev=1.0),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[0, 1, 2],
                ),
            ],
            scope=[0, 1, 2],
            weights=np.array([0.4, 0.6]),
        )

        result = likelihood(spn, data=np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
        self.assertAlmostEqual(result[0][0], 0.023358)

    def test_inference_2(self):
        spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        SumNode(
                            children=[
                                ProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                                ProductNode(
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
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        Gaussian(scope=[1], mean=0, stdev=1.0),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[0, 1, 2],
                ),
            ],
            scope=[0, 1, 2],
            weights=np.array([0.4, 0.6]),
        )

        result = log_likelihood(spn, data=np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
        self.assertAlmostEqual(result[0][0], -3.7568156)

    def test_inference_3(self):
        spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        SumNode(
                            children=[
                                ProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                                ProductNode(
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
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        Gaussian(scope=[1], mean=0, stdev=1.0),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[0, 1, 2],
                ),
            ],
            scope=[0, 1, 2],
            weights=np.array([0.4, 0.6]),
        )

        result = likelihood(spn, data=np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
        self.assertAlmostEqual(result[0][0], 0.09653235)

    def test_inference_4(self):
        spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        SumNode(
                            children=[
                                ProductNode(
                                    children=[
                                        Gaussian(scope=[1], mean=0, stdev=1.0),
                                        Gaussian(scope=[2], mean=0, stdev=1.0),
                                    ],
                                    scope=[1, 2],
                                ),
                                ProductNode(
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
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=0, stdev=1.0),
                        Gaussian(scope=[1], mean=0, stdev=1.0),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[0, 1, 2],
                ),
            ],
            scope=[0, 1, 2],
            weights=np.array([0.4, 0.6]),
        )

        result = log_likelihood(spn, data=np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
        self.assertAlmostEqual(result[0][0], -2.33787707)


if __name__ == "__main__":
    unittest.main()
