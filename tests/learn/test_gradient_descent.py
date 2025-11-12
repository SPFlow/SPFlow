import logging

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spflow.learn.gradient_descent import negative_log_likelihood_loss, train_gradient_descent
from spflow.meta import Scope
from spflow.modules import Module


# Define a DummyModel class for testing
class DummyModel(Module):
    @property
    def feature_to_scope(self) -> list[Scope]:
        return [Scope([0])]

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

    def log_likelihood(self, data, cache=None):
        return self(data)

    def sample(
        self,
        num_samples: int | None = None,
        data: torch.Tensor | None = None,
        is_mpe: bool = False,
        cache=None,
        sampling_ctx=None,
    ):
        raise NotImplementedError

    def expectation_maximization(self, data: torch.Tensor, cache=None):
        raise NotImplementedError

    def maximum_likelihood_estimation(
        self,
        data: torch.Tensor,
        weights: torch.Tensor | None = None,
        cache=None,
    ):
        raise NotImplementedError

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache=None):
        return self


@pytest.fixture
def model(device):
    return DummyModel().to(device)


@pytest.fixture
def dataloader(device):
    data = torch.randn(30, 1).to(device)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=3)


def test_negative_log_likelihood_loss(model, dataloader, device):
    data = next(iter(dataloader))[0]
    loss = negative_log_likelihood_loss(model, data)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar


def test_train_gradient_descent_basic(model, dataloader):
    initial_params = [p.clone() for p in model.parameters()]
    train_gradient_descent(model, dataloader, lr=0.01, epochs=1)
    for p, initial_p in zip(model.parameters(), initial_params):
        param_change = torch.abs(p - initial_p).max().item()
        assert param_change > 1e-6, f"Parameters should change during training (change: {param_change})"


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


@pytest.mark.parametrize("epochs", [1, 3])
def test_train_gradient_descent_multiple_epochs(model, dataloader, epochs):
    initial_loss = negative_log_likelihood_loss(model, next(iter(dataloader))[0])
    train_gradient_descent(model, dataloader, epochs=epochs)
    final_loss = negative_log_likelihood_loss(model, next(iter(dataloader))[0])
    assert final_loss < initial_loss  # loss should decrease after training
