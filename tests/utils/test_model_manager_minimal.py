from tests.fixtures import auto_set_test_seed
import unittest

from pytest import fixture

from spflow.meta.data import Scope
from spflow.modules import Product
from spflow.modules.leaf import Normal
from spflow.utils.model_manager import load_model, save_model
import torch


@fixture
def example_model() -> Product:
    return Product(inputs=Normal(Scope([0]), mean=torch.randn(1, 1), std=torch.rand(1, 1)))


@fixture
def tmp_model_file(tmp_path):
    return tmp_path / "test_model.pkl"


def test_save_model(tmp_model_file, example_model):
    save_model(example_model, tmp_model_file)
    assert tmp_model_file.exists()


def test_load_model(tmp_model_file, example_model):
    save_model(example_model, tmp_model_file)
    load_model(tmp_model_file)


if __name__ == "__main__":
    unittest.main()
