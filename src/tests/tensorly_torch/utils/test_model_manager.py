import os

import tensorly as tl

import unittest
from spflow.tensorly.utils.model_manager import load_model, save_model
from spflow.meta.data import Scope
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn import ProductNode, SumNode


tc = unittest.TestCase()
filename = "save.p"

def create_example_spn():
    spn = SumNode(
        children=[
            ProductNode(
                children=[
                    Gaussian(Scope([0])),
                    SumNode(
                        children=[
                            ProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                            ProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                        ],
                        weights=tl.tensor([0.3, 0.7]),
                    ),
                ],
            ),
            ProductNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0])),
                            Gaussian(Scope([1])),
                        ]
                    ),
                    Gaussian(Scope([2])),
                ]
            ),
        ],
        weights=tl.tensor([0.4, 0.6]),
    )
    return spn

def test_save_model(do_for_all_backends):
    # save model at filename
    save_model(create_example_spn(), filename)
    tc.assertTrue(os.path.exists(filename))

    # delete model
    if os.path.exists(filename):
        os.remove(filename)

def test_load_model(do_for_all_backends):
    # save mdoel
    example_spn = create_example_spn()
    save_model(example_spn, filename)
    # load the same model
    loaded_model = load_model(filename)

    # check if model attributes are equal
    for m1, m2 in zip(example_spn.modules(), loaded_model.modules()):
        tc.assertTrue(m1.__class__ == m2.__class__)
        tc.assertTrue(m1.scope == m2.scope)
        tc.assertTrue(m1.backend == m2.backend)
        if (isinstance(m1, SumNode)):
            tc.assertTrue(tl.all(m1.weights == m2.weights))

    # delete model
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    unittest.main()