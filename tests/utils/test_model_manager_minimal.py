import torch
from pytest import fixture

from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.products import Product
from spflow.utils.model_manager import load_model, save_model


@fixture
def example_model() -> Product:
    return Product(inputs=Normal(Scope([0]), loc=torch.randn(1, 1), scale=torch.rand(1, 1)))


@fixture
def tmp_model_file(tmp_path):
    return tmp_path / "test_model.pkl"


def test_save_model(tmp_model_file, example_model):
    save_model(example_model, tmp_model_file)
    assert tmp_model_file.exists()


def test_load_model(tmp_model_file, example_model):
    save_model(example_model, tmp_model_file)
    loaded = load_model(tmp_model_file)

    assert isinstance(loaded, type(example_model))
    assert loaded.scope.query == example_model.scope.query
    assert list(loaded.state_dict().keys()) == list(example_model.state_dict().keys())
    for key, expected in example_model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected)

    data = torch.randn(4, len(example_model.scope.query))
    torch.testing.assert_close(loaded.log_likelihood(data), example_model.log_likelihood(data))
