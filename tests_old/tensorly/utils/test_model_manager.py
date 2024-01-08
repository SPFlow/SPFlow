import unittest

from pytest import fixture
import numpy as np

from spflow.modules.module import log_likelihood
from spflow.utils.model_manager import load_model, save_model
from spflow.meta.data import Scope
from spflow.modules.node import Gaussian
from spflow.structure.spn import Node, ProductNode, SumNode
from spflow.utils import Tensor
from spflow.tensor import ops as tle


@fixture
def example_spn() -> SumNode:
    return Node(
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


@fixture
def tmp_model_file(tmp_path):
    return tmp_path / "test_model.pkl"


def test_save_model(do_for_all_backends, tmp_model_file, example_spn):
    save_model(example_spn, tmp_model_file)
    assert tmp_model_file.exists()


def test_load_model(do_for_all_backends, tmp_model_file, example_spn):
    save_model(example_spn, tmp_model_file)
    loaded_model = load_model(tmp_model_file)

    # check if model attributes are equal
    for m1, m2 in zip(example_spn.modules(), loaded_model.modules()):
        assert m1.__class__ == m2.__class__
        assert m1.scope == m2.scope
        assert m1.backend == m2.backend
        if isinstance(m1, SumNode):
            assert tl.all(m1.weights == m2.weights)


def test_load_model_inference(do_for_all_backends, tmp_model_file, example_spn):
    dummy_data = tl.tensor([[1.0, 0.0, 1.0]])
    ll_result = log_likelihood(example_spn, dummy_data)
    save_model(example_spn, tmp_model_file)
    loaded_model = load_model(tmp_model_file)
    loaded_ll_result = log_likelihood(loaded_model, dummy_data)
    assert np.allclose(tle.toNumpy(ll_result), tle.toNumpy(loaded_ll_result))


if __name__ == "__main__":
    unittest.main()
