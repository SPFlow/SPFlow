import torch
from pytest import fixture

from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.products import Product
from spflow.utils.model_manager import load_model, save_model


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
