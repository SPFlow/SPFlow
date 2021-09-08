import unittest
from numpy.random.mtrand import RandomState  # type: ignore
from spn.python.structure.nodes.leaves.parametric.parametric import Gaussian
import numpy as np
from spn.python.structure.nodes.node import SumNode, ProductNode
from spn.python.structure.nodes.sampling import sample_instances


class TestSampling(unittest.TestCase):
    def test_full_sampling(self):
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
                        ProductNode(
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

        result = sample_instances(
            spn, np.array([np.nan, np.nan, np.nan] * 5).reshape(-1, 3), RandomState(123)
        )
        self.assertTrue(
            np.allclose(
                result,
                np.array(
                    [
                        [-0.43435128, 0.3861864, -0.09470897],
                        [-0.67888615, 1.17582904, -1.25388067],
                        [2.20593008, 0.73736858, 1.49138963],
                        [2.18678609, 1.49073203, -0.638902],
                        [1.0040539, -0.93583387, -0.44398196],
                    ]
                ),
            )
        )

    def test_partial_sampling(self):
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
                        ProductNode(
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

        result = sample_instances(
            spn, np.array([np.nan, 0, 0] * 5).reshape(-1, 3), RandomState(123)
        )
        self.assertTrue(
            np.allclose(
                result,
                np.array(
                    [
                        [-0.09470897, 0.0, 0.0],
                        [-0.67888615, 0.0, 0.0],
                        [1.49138963, 0.0, 0.0],
                        [-0.638902, 0.0, 0.0],
                        [-0.44398196, 0.0, 0.0],
                    ]
                ),
            ),
        )

    def test_parameter_sampling(self):
        spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=1, stdev=1.0),
                        Gaussian(scope=[1], mean=2, stdev=2.0),
                    ],
                    scope=[0, 1],
                ),
                ProductNode(
                    children=[
                        Gaussian(scope=[0], mean=3, stdev=3.0),
                        Gaussian(scope=[1], mean=4, stdev=4.0),
                    ],
                    scope=[0, 1],
                ),
            ],
            scope=[0, 1],
            weights=np.array([0.3, 0.7]),
        )

        result = np.mean(
            sample_instances(
                spn, np.array([np.nan, np.nan] * 1000000).reshape(-1, 2), RandomState(123)
            ),
            axis=0,
        )

        self.assertTrue(
            np.allclose(result, np.array([0.3 * 1 + 0.7 * 3, 0.3 * 2 + 0.7 * 4]), atol=0.01),
        )


if __name__ == "__main__":
    unittest.main()
