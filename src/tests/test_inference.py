import unittest
from spflow.base.inference.module import likelihood, log_likelihood
from spflow.base.structure.nodes.leaves.parametric import Gaussian
from spflow.base.inference.nodes.leaves.parametric.parametric import (
    node_likelihood,
    node_log_likelihood,
)
import numpy as np
from spflow.base.structure.nodes import ISumNode, IProductNode, INode
from spflow.base.structure.network_type import SPN, set_network_type
from spflow.base.structure.nodes.node_module import (
    SumNode,
    ProductNode,
    LeafNode,
)
from spflow.base.learning.context import RandomVariableContext  # type: ignore


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

        result = likelihood(spn, np.array([1.0, 0.0, 1.0]).reshape(-1, 3), SPN())
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

        result = log_likelihood(spn, np.array([1.0, 0.0, 1.0]).reshape(-1, 3), SPN())
        self.assertAlmostEqual(result[0][0], -3.7568156)

    def test_inference_ll(self):
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

        result = likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3), SPN())
        self.assertAlmostEqual(result[0][0], 0.09653235)

    def test_inference_log_ll(self):
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

        result = log_likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3), SPN())
        self.assertAlmostEqual(result[0][0], -2.33787707)

    def test_not_implemented(self):
        class ITestNode(INode):
            def __init__(self, children, scope) -> None:
                super().__init__(children=children, scope=scope)

        with self.assertRaises(NotImplementedError):
            spn = ITestNode([], [0])
            probs = node_likelihood(spn, data=np.array([1.0]).reshape(-1, 1))

        with self.assertRaises(NotImplementedError):
            spn = ITestNode([], [0])
            probs = node_log_likelihood(spn, data=np.array([1.0]).reshape(-1, 1))

    def test_not_implemented_2(self):
        class ITestNode(INode):
            def __init__(self, children, scope) -> None:
                super().__init__(children=children, scope=scope)

        with self.assertRaises(NotImplementedError):
            spn = ITestNode([], [0])
            probs = likelihood(spn, data=np.array([1.0]).reshape(-1, 1))

        with self.assertRaises(NotImplementedError):
            spn = ITestNode([], [0])
            probs = log_likelihood(spn, data=np.array([1.0]).reshape(-1, 1))

    def test_inference_node_modules_log_ll(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian, Gaussian])

        with set_network_type(SPN()):
            spn = SumNode(
                children=[
                    ProductNode(
                        children=[
                            LeafNode(scope=[0], context=context),
                            SumNode(
                                children=[
                                    ProductNode(
                                        children=[
                                            LeafNode(scope=[1], context=context),
                                            LeafNode(scope=[2], context=context),
                                        ],
                                        scope=[1, 2],
                                    ),
                                    ProductNode(
                                        children=[
                                            LeafNode(scope=[1], context=context),
                                            LeafNode(scope=[2], context=context),
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
                                    LeafNode(scope=[0], context=context),
                                    LeafNode(scope=[1], context=context),
                                ],
                                scope=[0, 1],
                            ),
                            LeafNode(scope=[2], context=context),
                        ],
                        scope=[0, 1, 2],
                    ),
                ],
                scope=[0, 1, 2],
                weights=np.array([0.4, 0.6]),
            )

        result_node_module = log_likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
        result_nodes = log_likelihood(
            spn.output_nodes[0], np.array([np.nan, 0.0, 1.0]).reshape(-1, 3), spn.network_type
        )
        self.assertAlmostEqual(result_node_module[0][0], result_nodes[0][0])

    def test_inference_node_modules_ll(self):
        context = RandomVariableContext(parametric_types=[Gaussian, Gaussian, Gaussian])

        with set_network_type(SPN()):
            spn = SumNode(
                children=[
                    ProductNode(
                        children=[
                            LeafNode(scope=[0], context=context),
                            SumNode(
                                children=[
                                    ProductNode(
                                        children=[
                                            LeafNode(scope=[1], context=context),
                                            LeafNode(scope=[2], context=context),
                                        ],
                                        scope=[1, 2],
                                    ),
                                    ProductNode(
                                        children=[
                                            LeafNode(scope=[1], context=context),
                                            LeafNode(scope=[2], context=context),
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
                                    LeafNode(scope=[0], context=context),
                                    LeafNode(scope=[1], context=context),
                                ],
                                scope=[0, 1],
                            ),
                            LeafNode(scope=[2], context=context),
                        ],
                        scope=[0, 1, 2],
                    ),
                ],
                scope=[0, 1, 2],
                weights=np.array([0.4, 0.6]),
            )

        result_node_module = likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
        result_nodes = likelihood(
            spn.output_nodes[0], np.array([np.nan, 0.0, 1.0]).reshape(-1, 3), spn.network_type
        )
        self.assertAlmostEqual(
            result_nodes[0][0],
            result_node_module[0][0],
        )


if __name__ == "__main__":
    unittest.main()
