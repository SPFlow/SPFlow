import random
import unittest

import numpy as np
import torch

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import Categorical as BaseCategorical
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure import AutoLeaf, marginalize, toBase, toTorch
from spflow.torch.structure.spn import Categorical


class TestCategorical(unittest.TestCase):
    def test_initialization(self):

        # valid parameters for Categorical distribution: p \in R^n, \all p_i \in p: p_i \in [0,1], \sum_i p_i = 1

        categorical = Categorical(Scope([0]))
        categorical = Categorical(Scope([0]), k=3)
        self.assertTrue(torch.allclose(categorical.p, torch.tensor([1./3, 1./3, 1./3])))
        categorical = Categorical(Scope([0]), k=2, p=[0.3, 0.7])

        self.assertRaises(Exception, Categorical, Scope([0]), k=2, p=[1.0])
        self.assertRaises(Exception, Categorical, Scope([0]), k=2, p=[0.8, 0.8])
        self.assertRaises(Exception, Categorical, Scope([0]), k=1, p=[-0.1, 1.1])
        self.assertRaises(Exception, Categorical, Scope([0]), k=1, p=[0.1, 1.1])
        self.assertRaises(Exception, Categorical, Scope([0]), k=0)
        self.assertRaises(Exception, Categorical, Scope([0]), k=1, p=np.nan)
        self.assertRaises(Exception, Categorical, Scope([0]), k=1, p=[])
        self.assertRaises(Exception, Categorical, Scope([0]), k=1, p=[np.inf])
        self.assertRaises(Exception, Categorical, Scope([0]), k=1, p=np.inf)

        self.assertRaises(Exception, Categorical, Scope([]))
        self.assertRaises(Exception, Categorical, Scope([0, 1]))
        self.assertRaises(Exception, Categorical, Scope([0], [1]))


    def test_accept(self):

        # discrete meta type
        self.assertTrue(Categorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

        # Bernoulli feature type class
        self.assertTrue(Categorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Categorical])]))

        # Bernoulli feature type instance
        self.assertTrue(Categorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])]))

        # invalid feature type
        self.assertFalse(Categorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # conditional scope
        self.assertFalse(Categorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # multivariate signature
        self.assertFalse(
            Categorical.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )


    def test_initialization_from_signatures(self):

        categorical = Categorical.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Discrete])])
        self.assertTrue(categorical.k == torch.tensor(2))
        self.assertTrue(torch.allclose(categorical.p, torch.tensor([0.5, 0.5])))

        categorical = Categorical.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Categorical])])
        self.assertTrue(categorical.k == torch.tensor(2))
        self.assertTrue(torch.allclose(categorical.p, torch.tensor([0.5, 0.5])))

        categorical = Categorical.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=4, p=[0.1, 0.2, 0.3, 0.4])])])
        self.assertTrue(categorical.k == torch.tensor(4))
        self.assertTrue(torch.allclose(categorical.p, torch.tensor([0.1, 0.2, 0.3, 0.4])))


        # invalid feature type
        self.assertRaises(
            ValueError,
            Categorical.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Categorical.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Categorical.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )


    def test_autoleaf(self):

        self.assertTrue(AutoLeaf.is_registered(Categorical))

        self.assertEqual(Categorical, AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])])]))

        categorical = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=4, p=[0.1, 0.2, 0.3, 0.4])])])
        self.assertTrue(isinstance(categorical, Categorical))
        self.assertTrue(categorical.k, torch.tensor(4))
        self.assertTrue(torch.allclose(categorical.p, torch.tensor([0.1, 0.2, 0.3, 0.4])))


    def test_structural_marginalization(self):

        categorical = Categorical(Scope([0]), k=2, p=[0.3, 0.7])

        self.assertTrue(marginalize(categorical, [1]) is not None)
        self.assertTrue(marginalize(categorical, [0]) is None)


    def test_base_backend_conversion(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i/sum(p) for p_i in p]

        torch_categorical = Categorical(Scope([0]), k=k, p=p)
        base_categorical = BaseCategorical(Scope([0]), k=k, p=p)

        self.assertTrue(np.allclose([torch_categorical.get_params()[0]], [toBase(torch_categorical).get_params()[0]]))
        self.assertTrue(np.allclose([torch_categorical.get_params()[1]], [toBase(torch_categorical).get_params()[1]]))
        self.assertTrue(np.allclose([base_categorical.get_params()[0]], [toTorch(base_categorical).get_params()[0]]))
        self.assertTrue(np.allclose([base_categorical.get_params()[1]], [toTorch(base_categorical).get_params()[1]]))



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
