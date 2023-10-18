import random
import unittest
from typing import Callable

import numpy as np
import torch

from spflow.base.structure.general.nodes.leaves.parametric.cond_categorical import CondCategorical as BaseCondCategorical
from spflow.meta.data import Scope, FeatureContext, FeatureTypes
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure.spn import CondCategorical
from spflow.torch.structure import toBase, toTorch, AutoLeaf, marginalize


class TestCondCategorical(unittest.TestCase):
    def test_initialization(self):

        condCategorical = CondCategorical(Scope([0], [1]))
        self.assertTrue(condCategorical.cond_f is None)
        condCategorical = CondCategorical(Scope([0], [1]), cond_f=lambda data:{"k": 4, "p": [0.1, 0.2, 0.3, 0.4]})
        self.assertTrue(isinstance(condCategorical.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondCategorical, Scope([]))
        self.assertRaises(Exception, CondCategorical, Scope([0, 1], [2]))
        self.assertRaises(Exception, CondCategorical, Scope([0]))


    def test_retrieve_params(self):

        # valid parameters for Categorical distribution: p \in R^n, \all p_i \in p: p_i \in [0,1], \sum_i p_i = 1

        condCategorical = CondCategorical(Scope([0], [1]))

        condCategorical.set_cond_f(lambda data: {"k": 2, "p": [0.3, 0.7]})
        k, p = condCategorical.retrieve_params(np.array([[1]]), DispatchContext())
        self.assertEqual(k, torch.tensor(2))
        self.assertTrue(torch.allclose(p, torch.tensor([0.3, 0.7])))

        condCategorical.set_cond_f(lambda data: {"k": 1, "p": [0.5]})
        self.assertRaises(Exception, condCategorical.retrieve_params, np.array([[1]]), DispatchContext())
        condCategorical.set_cond_f(lambda data: {"k": 1, "p": [1.1]})
        self.assertRaises(Exception, condCategorical.retrieve_params, np.array([[1]]), DispatchContext())
        condCategorical.set_cond_f(lambda data: {"k": 2, "p": [0.8, 0.8]})
        self.assertRaises(Exception, condCategorical.retrieve_params, np.array([[1]]), DispatchContext())
        condCategorical.set_cond_f(lambda data: {"k": 2, "p": [0.7, -0.1]})
        self.assertRaises(Exception, condCategorical.retrieve_params, np.array([[1]]), DispatchContext())



    def test_accept(self):
        
        # discrete meta type
        self.assertTrue(CondCategorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # Bernoulli feature type class
        self.assertTrue(CondCategorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical])]))

        # Bernoulli feature type instance
        self.assertTrue(-CondCategorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])])]))

        # invalid feature type
        self.assertFalse(CondCategorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

        # non-conditional scope
        self.assertFalse(CondCategorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

        # multivariate signature
        self.assertFalse(
            CondCategorical.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    
    
    def test_initialization_from_signatures(self):

        CondCategorical.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])])
        CondCategorical.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical])])
        CondCategorical.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical(k=4, p=[0.1, 0.2, 0.3, 0.4])])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondCategorical.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondCategorical.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondCategorical.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )


    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondCategorical))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondCategorical,
            AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        categorical = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical])])
        self.assertTrue(isinstance(categorical, CondCategorical))


    
    def test_structural_marginalization(self):

        categorical = CondCategorical(Scope([0], [1]))

        self.assertTrue(marginalize(categorical, [1]) is not None)
        self.assertTrue(marginalize(categorical, [0]) is None)


    def test_base_backend_conversion(self):

        k = 4
        p = [random.random() for _ in range(k)]
        p = [p_i/sum(p) for p_i in p]

        torch_categorical = CondCategorical(Scope([0], [1]), lambda data: {"k": k, "p": p})
        base_categorical = BaseCondCategorical(Scope([0], [1]), lambda data: {"k": k, "p": p})

        self.assertTrue(np.all(torch_categorical.scopes_out == toBase(torch_categorical).scopes_out))
        self.assertTrue(np.all(base_categorical.scopes_out == toTorch(base_categorical).scopes_out))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()


