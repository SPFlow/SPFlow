import unittest
from numpy.random.mtrand import RandomState  # type: ignore
import numpy as np
from spflow.base.structure.nodes.node import ISumNode, IProductNode
from spflow.base.structure.network_type import SPN
from spflow.base.structure.rat.rat_spn import RatSpn
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
from spflow.base.structure.rat.region_graph import random_region_graph
from spflow.base.sampling.rat.rat_spn import sample_instances
from spflow.base.learning.context import Context  # type: ignore
from spflow.base.structure.nodes.leaves.parametric import Gaussian


class TestSampling(unittest.TestCase):
    def test_full_sampling(self):
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

        result = sample_instances(
            SPN(), spn, np.array([np.nan, np.nan, np.nan] * 5).reshape(-1, 3), RandomState(123)
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

        result = sample_instances(
            SPN(), spn, np.array([np.nan, 0, 0] * 5).reshape(-1, 3), RandomState(123)
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

    def test_parameter_sampling_nodes(self):
        spn = ISumNode(
            children=[
                IProductNode(
                    children=[
                        Gaussian(scope=[0], mean=1, stdev=1.0),
                        Gaussian(scope=[1], mean=2, stdev=2.0),
                    ],
                    scope=[0, 1],
                ),
                IProductNode(
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
                SPN(), spn, np.array([np.nan, np.nan] * 1000000).reshape(-1, 2), RandomState(123)
            ),
            axis=0,
        )

        self.assertTrue(
            np.allclose(result, np.array([0.3 * 1 + 0.7 * 3, 0.3 * 2 + 0.7 * 4]), atol=0.01),
        )

    def test_parameter_sampling_rat_module(self):

        region_graph = random_region_graph(X=set(range(0, 2)), depth=1, replicas=1)
        context = Context(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )
        rat_spn_module = RatSpn(region_graph, 1, 1, 1, context)
        _isvalid_spn(rat_spn_module.output_nodes[0])

        result = np.mean(
            sample_instances(
                rat_spn_module,
                np.array([np.nan, np.nan] * 1000000).reshape(-1, 2),
                RandomState(123),
            ),
            axis=0,
        )

        self.assertTrue(
            np.allclose(
                result,
                np.array([rat_spn_module.nodes[0].mean, rat_spn_module.nodes[1].mean]),
                atol=0.01,
            ),
        )


if __name__ == "__main__":
    unittest.main()
