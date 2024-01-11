"""TODO: A dummy test case that shall be replaced by the proper one in `tests_old`."""

import unittest

from pytest import fixture

from spflow.meta.data import Scope
from spflow.modules.node import Node, ProductNode
from spflow.modules.node.leaf import Gaussian
from spflow.utils.model_manager import load_model, save_model
from ..fixtures import backend_auto


@fixture
def example_spn() -> Node:
    return ProductNode(
        inputs=[
            Gaussian(Scope([0])),
        ],
    )


@fixture
def tmp_model_file(tmp_path):
    return tmp_path / "test_model.pkl"


def test_save_model(backend_auto, tmp_model_file, example_spn):
    save_model(example_spn, tmp_model_file)
    assert tmp_model_file.exists()


def test_load_model(backend_auto, tmp_model_file, example_spn):
    save_model(example_spn, tmp_model_file)
    load_model(tmp_model_file)


if __name__ == "__main__":
    unittest.main()
