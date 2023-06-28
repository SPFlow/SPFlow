import unittest
from typing import Callable

import numpy as np
import torch

from spflow.base.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussian as BaseCondMultivariateGaussian,
)
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.spn import CondMultivariateGaussian as TorchCondMultivariateGaussian
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussian
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)
from spflow.torch.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import (
    #CondMultivariateGaussian,
    marginalize,
    toBase,
    toTorch,
)
from spflow.torch.structure.spn.nodes.product_node import ProductNode


class TestMultivariateGaussian(unittest.TestCase):
    def test_initialization(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0], [1]))
        self.assertTrue(multivariate_gaussian.cond_f is None)
        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0, 1], [2]),
            lambda x: {"mean": torch.zeros(2), "cov": torch.eye(2)},
        )
        self.assertTrue(isinstance(multivariate_gaussian.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondMultivariateGaussian, Scope([]))
        self.assertRaises(Exception, CondMultivariateGaussian, Scope([0, 1]))

    def test_retrieve_params(self):

        # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))

        # mean contains inf and mean contains nan
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": torch.tensor([0.0, float("inf")]),
                "cov": torch.eye(2),
            }
        )
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": torch.tensor([-float("inf"), 0.0]),
                "cov": torch.eye(2),
            }
        )
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": torch.tensor([0.0, float("na")]),
                "cov": torch.eye(2),
            }
        )
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )

        # mean vector of wrong shape
        multivariate_gaussian.set_cond_f(lambda data: {"mean": torch.zeros(3), "cov": torch.eye(2)})
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(lambda data: {"mean": torch.zeros(1, 1, 2), "cov": torch.eye(2)})
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )

        # covariance matrix of wrong shape
        M = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        multivariate_gaussian.set_cond_f(lambda data: {"mean": torch.zeros(2), "cov": M})
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(lambda data: {"mean": torch.zeros(2), "cov": M.T})
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(lambda data: {"mean": torch.zeros(2), "cov": torch.zeros(3)})
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )

        # covariance matrix not symmetric positive semi-definite
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": torch.zeros(2),
                "cov": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            }
        )
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(lambda data: {"mean": torch.zeros(2), "cov": -torch.eye(2)})
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )

        # covariance matrix containing inf or nan
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": torch.zeros(2),
                "cov": torch.tensor([[float("inf"), 0], [0, float("inf")]]),
            }
        )
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": torch.zeros(2),
                "cov": torch.tensor([[float("nan"), 0], [0, float("nan")]]),
            }
        )
        self.assertRaises(
            Exception,
            multivariate_gaussian.retrieve_params,
            torch.tensor([[1.0]]),
            DispatchContext(),
        )

        # initialize using lists
        multivariate_gaussian.set_cond_f(lambda data: {"mean": [0.0, 0.0], "cov": [[1.0, 0.0], [0.0, 1.0]]})
        mean, cov, cov_tril = multivariate_gaussian.retrieve_params(torch.tensor([[1.0]]), DispatchContext())
        self.assertTrue(cov_tril is None)
        self.assertTrue(torch.all(mean == torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.all(cov == torch.tensor([[1.0, 0.0], [0.0, 1.0]])))

        # initialize using numpy arrays
        multivariate_gaussian.set_cond_f(lambda data: {"mean": np.zeros(2), "cov": np.eye(2)})
        mean, cov, cov_tril = multivariate_gaussian.retrieve_params(torch.tensor([[1.0]]), DispatchContext())
        self.assertTrue(cov_tril is None)
        self.assertTrue(torch.all(mean == torch.zeros(2)))
        self.assertTrue(torch.all(cov == torch.eye(2)))

    def test_accept(self):

        # continuous meta types
        self.assertTrue(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

        # Gaussian feature type class
        self.assertTrue(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    )
                ]
            )
        )

        # Gaussian feature type instance
        self.assertTrue(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [
                            FeatureTypes.Gaussian(0.0, 1.0),
                            FeatureTypes.Gaussian(0.0, 1.0),
                        ],
                    )
                ]
            )
        )

        # continuous meta and Gaussian feature types
        self.assertTrue(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Gaussian],
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Discrete, FeatureTypes.Continuous],
                    )
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondMultivariateGaussian.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
        CondMultivariateGaussian.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                )
            ]
        )
        CondMultivariateGaussian.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [
                        FeatureTypes.Gaussian(-1.0, 1.5),
                        FeatureTypes.Gaussian(1.0, 0.5),
                    ],
                )
            ]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondMultivariateGaussian.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Continuous],
                )
            ],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondMultivariateGaussian.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondMultivariateGaussian))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondMultivariateGaussian,
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    )
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        multivariate_gaussian = AutoLeaf(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [
                        FeatureTypes.Gaussian(mean=-1.0, std=1.5),
                        FeatureTypes.Gaussian(mean=1.0, std=0.5),
                    ],
                )
            ]
        )
        self.assertTrue(isinstance(multivariate_gaussian, TorchCondMultivariateGaussian))

    def test_structural_marginalization(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [3]))

        self.assertTrue(
            isinstance(
                marginalize(multivariate_gaussian, [2]),
                TorchCondMultivariateGaussian,
            )
        )
        self.assertTrue(isinstance(marginalize(multivariate_gaussian, [1]), CondGaussian))
        self.assertTrue(marginalize(multivariate_gaussian, [0, 1]) is None)

    def test_base_backend_conversion(self):

        torch_multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1, 2], [3]))
        node_multivariate_gaussian = BaseCondMultivariateGaussian(Scope([0, 1, 2], [3]))

        # check conversion from torch to python
        self.assertTrue(
            np.all(torch_multivariate_gaussian.scopes_out == toBase(torch_multivariate_gaussian).scopes_out)
        )
        # check conversion from python to torch
        self.assertTrue(np.all(node_multivariate_gaussian.scopes_out == toTorch(node_multivariate_gaussian).scopes_out))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
