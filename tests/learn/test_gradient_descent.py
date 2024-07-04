import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock
import logging
from spflow.modules.module import Module
from spflow.learn.gradient_descent import negative_log_likelihood_loss, train_gradient_descent
from spflow.meta.dispatch import dispatch


# Define a DummyModel class for testing
class DummyModel(Module):
    @property
    def out_channels(self) -> int:
        return 1

    @property
    def out_features(self) -> int:
        return 1

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Mock the log_likelihood function
@dispatch(memoize=True)
def log_likelihood(model: DummyModel, data: torch.Tensor):
    return model(data)


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def dataloader():
    data = torch.randn(100, 1)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=10)


def test_negative_log_likelihood_loss(model, dataloader):
    data = next(iter(dataloader))[0]
    loss = negative_log_likelihood_loss(model, data)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar


def test_train_gradient_descent_basic(model, dataloader):
    initial_params = [p.clone() for p in model.parameters()]
    train_gradient_descent(model, dataloader, epochs=1)
    for p, initial_p in zip(model.parameters(), initial_params):
        assert not torch.allclose(p, initial_p)  # parameters should have changed


def test_train_gradient_descent_custom_optimizer(model, dataloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_gradient_descent(model, dataloader, epochs=1, optimizer=optimizer)
    assert isinstance(model.linear.weight.grad, torch.Tensor)  # gradients should be computed


def test_train_gradient_descent_custom_loss_fn(model, dataloader):
    def custom_loss_fn(model, data):
        return torch.sum(model(data) ** 2)

    train_gradient_descent(model, dataloader, epochs=1, loss_fn=custom_loss_fn)
    assert isinstance(model.linear.weight.grad, torch.Tensor)  # gradients should be computed


def test_train_gradient_descent_callbacks(model, dataloader):
    batch_calls = []
    epoch_calls = []

    def batch_callback(loss, step):
        batch_calls.append((loss.item(), step))

    def epoch_callback(losses, epoch):
        epoch_calls.append((len(losses), epoch))

    train_gradient_descent(
        model, dataloader, epochs=2, callback_batch=batch_callback, callback_epoch=epoch_callback
    )

    assert len(batch_calls) == 20  # 2 epochs * 10 batches
    assert len(epoch_calls) == 2  # 2 epochs
    assert epoch_calls[0][0] == 10  # 10 losses per epoch
    assert epoch_calls[1][1] == 1  # second epoch index


def test_train_gradient_descent_verbose(model, dataloader, caplog):
    caplog.set_level(logging.INFO)
    train_gradient_descent(model, dataloader, epochs=2, verbose=True)
    assert len(caplog.records) == 2  # 2 log messages for 2 epochs
    for record in caplog.records:
        assert "Epoch" in record.message
        assert "Loss:" in record.message


@pytest.mark.parametrize("epochs", [1, 5, 10])
def test_train_gradient_descent_multiple_epochs(model, dataloader, epochs):
    initial_loss = negative_log_likelihood_loss(model, next(iter(dataloader))[0])
    train_gradient_descent(model, dataloader, epochs=epochs)
    final_loss = negative_log_likelihood_loss(model, next(iter(dataloader))[0])
    assert final_loss < initial_loss  # loss should decrease after training


def test_train_gradient_descent_empty_dataloader():
    model = DummyModel()
    empty_dataloader = DataLoader(TensorDataset(torch.Tensor([])))
    train_gradient_descent(model, empty_dataloader, epochs=1)  # should not raise an error


def test_train_gradient_descent_nan_loss(model, dataloader):
    def nan_loss_fn(model, data):
        return torch.tensor(float("nan"))

    with pytest.raises(RuntimeError):  # or whatever exception you expect
        train_gradient_descent(model, dataloader, epochs=1, loss_fn=nan_loss_fn)
