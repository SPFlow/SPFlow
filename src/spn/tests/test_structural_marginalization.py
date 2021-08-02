import unittest
from spn.base.nodes.structural_marginalization import marginalize
from spn.base.nodes.leaves.parametric.parametric import Gaussian
import numpy as np
from spn.base.nodes.node import (
    SumNode,
    ProductNode,
)


class TestStructuralMarginalization(unittest.TestCase):
    def test_structural_marginalization_1(self):
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
        spn_marg = marginalize(spn, [1, 2])

        spn_marg_correct = SumNode(
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
                ProductNode(
                    children=[
                        Gaussian(scope=[1], mean=0, stdev=1.0),
                        Gaussian(scope=[2], mean=0, stdev=1.0),
                    ],
                    scope=[1, 2],
                ),
            ],
            scope=[1, 2],
            weights=np.array([0.6, 0.12, 0.28]),
        )

        self.assertTrue(spn_marg.equals(spn_marg_correct))

    def test_structural_marginalization_2(self):
        spn = spn = SumNode(
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
        spn_marg = marginalize(spn, [])

        spn_marg_correct = None

        self.assertEqual(spn_marg, spn_marg_correct)


if __name__ == "__main__":
    unittest.main()
