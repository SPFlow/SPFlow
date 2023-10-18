import unittest
import numpy as np
from typing import Callable

from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn import CondCategorical, marginalize
from spflow.meta.data import Scope, FeatureContext, FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext

class TestCondCategorical(unittest.TestCase):

    def test_initialization(self):

        condCategorical = CondCategorical(Scope([0], [1]))
        self.assertTrue(condCategorical.cond_f is None)
        condCategorical = CondCategorical(Scope([0], [1]), lambda x: {"k": 2, "p": [0.5, 0.5]})
        self.assertTrue(isinstance(condCategorical.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondCategorical, Scope([]))
        self.assertRaises(Exception, CondCategorical, Scope([0]))
        self.assertRaises(Exception, CondCategorical, Scope([0, 1]), [2])


    def test_retrieve_params(self):

        # valid parameters for Categorical distribution: p \in R^n, \all p_i \in p: p_i \in [0,1], \sum_i p_i = 1

        condCategorical = CondCategorical(Scope([0], [1]))

        condCategorical.set_cond_f(lambda data: {"k": 1, "p": [1.0]})
        k, p = condCategorical.retrieve_params(np.array([[1.0]]), DispatchContext())
        self.assertEqual(k, 1)
        self.assertEqual(p, [1.0])

        condCategorical.set_cond_f(lambda data: {"k": 4, "p": [0.1, 0.2, 0.3, 0.4]})
        k, p = condCategorical.retrieve_params(np.array([[1.0]]), DispatchContext())
        self.assertEqual(k, 4)
        self.assertEqual(p, [0.1, 0.2, 0.3, 0.4])

        # p_i < 0, p_i > 1
        condCategorical.set_cond_f(lambda data: {"k": 1, "p": np.nextafter(1.0, 2.0)})
        self.assertRaises(ValueError, condCategorical.retrieve_params, np.array([[1.0]]), DispatchContext())
        
        condCategorical.set_cond_f(lambda data: {"k": 2, "p": np.nextafter(0.0, -1.0)})
        self.assertRaises(ValueError, condCategorical.retrieve_params, np.array([[1.0]]), DispatchContext())
    
        # \sum p_i != 1
        condCategorical.set_cond_f(lambda data: {"k": 2, "p": [0.5, 0.4]})
        self.assertRaises(ValueError, condCategorical.retrieve_params, np.array([[1.0]]), DispatchContext())

        condCategorical.set_cond_f(lambda data: {"k": 2, "p": [0.5, 0.6]})
        self.assertRaises(ValueError, condCategorical.retrieve_params, np.array([[1.0]]), DispatchContext())

        # p not a list/array
        condCategorical.set_cond_f(lambda data: {"k": 1, "p": 1.0})
        self.assertRaises(ValueError, condCategorical.retrieve_params, np.array([[1.0]]), DispatchContext())

        # k and |p| not matching
        condCategorical.set_cond_f(lambda data: {"k": 2, "p": [1.0]})
        self.assertRaises(ValueError, condCategorical.retrieve_params, np.array([[1.0]]), DispatchContext())

        # p = inf, p = nan
        condCategorical.set_cond_f(lambda data: {"k": 2, "p": [1.0, np.nan]})
        self.assertRaises(ValueError, condCategorical.retrieve_params, np.array([[1.0]]), DispatchContext())

        condCategorical.set_cond_f(lambda data: {"k": 2, "p": [1.0, np.inf]})
        self.assertRaises(ValueError, condCategorical.retrieve_params, np.array([[1.0]]), DispatchContext())


    def test_accept(self):

        # discrete meta type
        self.assertTrue(CondCategorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # Categorical feature type class
        self.assertTrue(CondCategorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical])]))

        # Categorical feature type instance
        self.assertTrue(CondCategorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical(k=2, p=[0.5, 0,5])])]))

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
        CondCategorical.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])])

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
        condCategorical = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical])])
        self.assertTrue(isinstance(condCategorical, CondCategorical))

    def test_conditional_marginalization(self):

        condCategorical = CondCategorical(Scope([0], [2]))

        self.assertTrue(marginalize(condCategorical, [1]) is not None)
        self.assertTrue(marginalize(condCategorical, [0]) is None)


if __name__ == "__main__":
    unittest.main()

    
