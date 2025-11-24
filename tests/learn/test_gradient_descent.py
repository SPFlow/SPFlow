import logging

import numpy as np
from spflow.interfaces.classifier import Classifier
from spflow.utils.cache import cached
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spflow.learn.gradient_descent import (
    TrainingMetrics,
    _extract_batch_data,
    _run_validation_epoch,
    nll_loss,
    negative_log_likelihood_loss,
    train_gradient_descent,
)
from spflow.meta import Scope
from spflow.modules.base import Module


# Define a DummyModel class for testing
class DummyModel(Module):
    @property
    def feature_to_scope(self) -> np.array:
        return np.array([Scope([0])]).view(1, 1)

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

    @cached
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


class ClassificationModel(Module, Classifier):
    """Minimal classification model with log-prob outputs for testing."""

    def __init__(self, num_features: int = 2, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
        self.posterior_calls = 0
        self.likelihood_calls = 0
        self.scope = Scope(list(range(num_features)))

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([idx]) for idx in range(self.out_features)]).view(-1, 1)

    @property
    def out_channels(self) -> int:
        return self.linear.out_features

    @property
    def out_features(self) -> int:
        return self.linear.in_features

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        return self.log_posterior(data).exp()

    def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:
        self.likelihood_calls += 1
        logits = self.linear(data)
        return torch.log_softmax(logits, dim=1)

    def log_posterior(self, data: torch.Tensor, log_prior=None, cache=None) -> torch.Tensor:
        self.posterior_calls += 1
        return self.log_likelihood(data, cache=cache)

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
def classification_model(device) -> ClassificationModel:
    return ClassificationModel().to(device)


@pytest.fixture
def classification_dataloader(device):
    torch.manual_seed(0)
    features = torch.randn(12, 2, device=device)
    labels = torch.randint(0, 3, (12,), device=device)
    return DataLoader(TensorDataset(features, labels), batch_size=4)


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


def test_train_gradient_descent_custom_scheduler(model, dataloader):
    """Custom scheduler should be stepped once per epoch."""

    class CountingScheduler:
        def __init__(self):
            self.calls = 0

        def step(self):
            self.calls += 1

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = CountingScheduler()

    epochs = 3
    train_gradient_descent(
        model,
        dataloader,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=negative_log_likelihood_loss,
    )

    assert scheduler.calls == epochs  # scheduler.step should be called once per epoch


def test_extract_batch_data_validation():
    """Ensure batch extraction handles classification and regression formats."""
    data = torch.randn(3, 2)
    targets = torch.tensor([0, 1, 2])

    out_data, out_targets = _extract_batch_data((data, targets), is_classification=True)
    torch.testing.assert_close(out_data, data)
    torch.testing.assert_close(out_targets, targets)

    reg_data, reg_targets = _extract_batch_data((data, targets), is_classification=False)
    torch.testing.assert_close(reg_data, data)
    assert reg_targets is None

    with pytest.raises(ValueError):
        _extract_batch_data(data, is_classification=True)

    with pytest.raises(ValueError):
        _extract_batch_data((data, targets, targets), is_classification=False)


def test_train_gradient_descent_requires_classifier_for_classification_mode(model, dataloader):
    """Ensure TypeError is raised when using is_classification=True with non-Classifier model."""
    with pytest.raises(TypeError, match="model must be a Classifier instance when is_classification=True"):
        train_gradient_descent(
            model,  # DummyModel doesn't implement Classifier
            dataloader,
            epochs=1,
            is_classification=True,
        )


def test_train_gradient_descent_classification_mode(classification_model, classification_dataloader):
    """Train in classification mode and ensure posterior path is used."""
    initial_params = [p.clone() for p in classification_model.parameters()]

    train_gradient_descent(
        classification_model,
        classification_dataloader,
        epochs=2,
        is_classification=True,
        lr=0.05,
    )

    assert classification_model.posterior_calls > 0
    assert classification_model.likelihood_calls > 0
    for param, initial in zip(classification_model.parameters(), initial_params):
        assert not torch.allclose(param, initial)


def test_run_validation_epoch_classification(classification_model, classification_dataloader):
    """Validation loop computes losses and accuracy for classification batches."""
    metrics = TrainingMetrics()
    val_loss = _run_validation_epoch(
        classification_model,
        classification_dataloader,
        nll_loss,
        metrics,
        is_classification=True,
        callback_batch=None,
    )

    assert isinstance(val_loss, torch.Tensor)
    assert metrics.val_total == 12
    assert metrics.validation_steps == len(classification_dataloader)
    assert metrics.get_val_accuracy() >= 0.0


def test_train_gradient_descent_classification_with_validation(classification_model, device):
    """Training with validation dataloader should hit classification validation logic."""
    torch.manual_seed(1)
    train_loader = DataLoader(
        TensorDataset(torch.randn(12, 2, device=device), torch.randint(0, 3, (12,), device=device)),
        batch_size=4,
    )
    val_loader = DataLoader(
        TensorDataset(torch.randn(8, 2, device=device), torch.randint(0, 3, (8,), device=device)),
        batch_size=4,
    )

    train_gradient_descent(
        classification_model,
        train_loader,
        epochs=1,
        is_classification=True,
        lr=0.05,
        validation_dataloader=val_loader,
    )

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    assert classification_model.posterior_calls == train_batches
    assert classification_model.likelihood_calls == train_batches * 2 + val_batches


@pytest.fixture
def training_metrics():
    """Create a fresh TrainingMetrics instance for each test."""
    return TrainingMetrics()


@pytest.fixture
def sample_tensors():
    """Provide sample tensors for testing."""
    return {
        "loss_small": torch.tensor(0.5),
        "loss_medium": torch.tensor(1.5),
        "loss_large": torch.tensor(2.5),
        "predictions_4": torch.tensor([0, 1, 1, 0]),
        "targets_4": torch.tensor([0, 0, 1, 0]),
        "predictions_3": torch.tensor([1, 0, 1]),
        "targets_3": torch.tensor([1, 1, 1]),
        "predictions_5": torch.tensor([0, 1, 1, 0, 1]),
        "targets_5": torch.tensor([0, 0, 1, 0, 1]),
    }


class TestTrainingMetrics:
    """Comprehensive test suite for TrainingMetrics class.

    Tests cover initialization, batch updates, accuracy calculations,
    metric resets, and edge cases for both regression and classification tasks.
    """

    def test_initialization_with_default_values(self, training_metrics):
        """Test TrainingMetrics initializes with correct default values."""
        assert training_metrics.train_losses == []
        assert training_metrics.val_losses == []
        assert training_metrics.train_correct == 0
        assert training_metrics.train_total == 0
        assert training_metrics.val_correct == 0
        assert training_metrics.val_total == 0
        assert training_metrics.training_steps == 0
        assert training_metrics.validation_steps == 0

    @pytest.mark.parametrize("loss_value", [0.1, 1.0, 5.5, 10.0])
    def test_update_train_batch_regression_only(self, training_metrics, loss_value):
        """Test updating training batch metrics for regression tasks with various loss values."""
        loss = torch.tensor(loss_value)

        training_metrics.update_train_batch(loss)

        assert len(training_metrics.train_losses) == 1
        assert torch.equal(training_metrics.train_losses[0], loss)
        assert training_metrics.training_steps == 1
        assert training_metrics.train_total == 0
        assert training_metrics.train_correct == 0

    @pytest.mark.parametrize(
        "predictions,targets,expected_correct,expected_total",
        [
            ([0, 1, 1, 0], [0, 0, 1, 0], 3, 4),  # 3/4 correct
            ([1, 0, 1], [1, 1, 1], 2, 3),  # 2/3 correct
            ([0, 1, 2], [0, 1, 2], 3, 3),  # 3/3 correct
            ([0, 1, 2], [2, 1, 0], 1, 3),  # 1/3 correct
            ([], [], 0, 0),  # Empty tensors
        ],
    )
    def test_update_train_batch_classification_accuracy(
        self, training_metrics, predictions, targets, expected_correct, expected_total
    ):
        """Test classification accuracy calculation with various prediction scenarios."""
        loss = torch.tensor(1.0)
        pred_tensor = torch.tensor(predictions) if predictions else torch.tensor([])
        target_tensor = torch.tensor(targets) if targets else torch.tensor([])

        training_metrics.update_train_batch(loss, pred_tensor, target_tensor)

        assert len(training_metrics.train_losses) == 1
        assert training_metrics.training_steps == 1
        assert training_metrics.train_total == expected_total
        assert training_metrics.train_correct == expected_correct

    def test_update_train_batch_classification_with_partial_predictions(self, training_metrics):
        """Test classification update when only one of predicted/targets is provided."""
        loss = torch.tensor(1.0)
        predictions = torch.tensor([0, 1, 0])

        # Only predictions provided, no targets
        training_metrics.update_train_batch(loss, predictions, None)

        assert training_metrics.train_total == 0
        assert training_metrics.train_correct == 0
        assert training_metrics.training_steps == 1

    @pytest.mark.parametrize("loss_value", [0.2, 2.0, 7.5])
    def test_update_val_batch_regression_only(self, training_metrics, loss_value):
        """Test updating validation batch metrics for regression tasks."""
        loss = torch.tensor(loss_value)

        training_metrics.update_val_batch(loss)

        assert len(training_metrics.val_losses) == 1
        assert torch.equal(training_metrics.val_losses[0], loss)
        assert training_metrics.validation_steps == 1
        assert training_metrics.val_total == 0
        assert training_metrics.val_correct == 0

    @pytest.mark.parametrize(
        "predictions,targets,expected_correct,expected_total",
        [
            ([1, 0, 1], [1, 1, 1], 2, 3),  # 2/3 correct
            ([0, 1, 0], [0, 1, 0], 3, 3),  # 3/3 correct
            ([1, 1, 1], [0, 0, 0], 0, 3),  # 0/3 correct
        ],
    )
    def test_update_val_batch_classification_accuracy(
        self, training_metrics, predictions, targets, expected_correct, expected_total
    ):
        """Test validation classification accuracy calculation."""
        loss = torch.tensor(0.8)
        pred_tensor = torch.tensor(predictions)
        target_tensor = torch.tensor(targets)

        training_metrics.update_val_batch(loss, pred_tensor, target_tensor)

        assert len(training_metrics.val_losses) == 1
        assert training_metrics.validation_steps == 1
        assert training_metrics.val_total == expected_total
        assert training_metrics.val_correct == expected_correct

    @pytest.mark.parametrize(
        "correct,total,expected_accuracy",
        [
            (0, 0, 0.0),  # No samples
            (0, 10, 0.0),  # All incorrect
            (5, 10, 50.0),  # Half correct
            (8, 10, 80.0),  # 80% correct
            (10, 10, 100.0),  # All correct
            (1, 3, 33.33333333333333),  # Repeating decimal
        ],
    )
    def test_get_train_accuracy_various_scenarios(self, training_metrics, correct, total, expected_accuracy):
        """Test training accuracy calculation with various correct/total combinations."""
        training_metrics.train_correct = correct
        training_metrics.train_total = total

        accuracy = training_metrics.get_train_accuracy()
        assert accuracy == pytest.approx(expected_accuracy)

    @pytest.mark.parametrize(
        "correct,total,expected_accuracy",
        [
            (0, 0, 0.0),  # No samples
            (3, 10, 30.0),  # 30% correct
            (7, 7, 100.0),  # All correct
            (2, 8, 25.0),  # 25% correct
        ],
    )
    def test_get_val_accuracy_various_scenarios(self, training_metrics, correct, total, expected_accuracy):
        """Test validation accuracy calculation with various correct/total combinations."""
        training_metrics.val_correct = correct
        training_metrics.val_total = total

        accuracy = training_metrics.get_val_accuracy()
        assert accuracy == pytest.approx(expected_accuracy)

    def test_reset_epoch_metrics_comprehensive_reset(self, training_metrics, sample_tensors):
        """Test that reset_epoch_metrics properly resets all per-epoch metrics while preserving cumulative ones."""
        # Populate all metrics with data
        training_metrics.train_losses = [sample_tensors["loss_small"], sample_tensors["loss_medium"]]
        training_metrics.val_losses = [sample_tensors["loss_large"]]
        training_metrics.train_correct = 5
        training_metrics.train_total = 10
        training_metrics.val_correct = 3
        training_metrics.val_total = 8
        training_metrics.training_steps = 5
        training_metrics.validation_steps = 2

        # Reset epoch metrics
        training_metrics.reset_epoch_metrics()

        # Verify all per-epoch metrics are reset to initial state
        assert training_metrics.train_losses == []
        assert training_metrics.val_losses == []
        assert training_metrics.train_correct == 0
        assert training_metrics.train_total == 0
        assert training_metrics.val_correct == 0
        assert training_metrics.val_total == 0

        # Verify cumulative metrics are preserved (critical bug fix verification)
        assert training_metrics.training_steps == 5
        assert training_metrics.validation_steps == 2

    def test_multiple_batches_accuracy_accumulation_across_batches(self, training_metrics):
        """Test accuracy calculation across multiple training batches."""
        # First batch: 4/5 correct
        training_metrics.update_train_batch(
            torch.tensor(1.0), torch.tensor([0, 1, 1, 0, 1]), torch.tensor([0, 0, 1, 0, 1])
        )

        # Second batch: 2/4 correct
        training_metrics.update_train_batch(
            torch.tensor(1.5), torch.tensor([1, 0, 1, 0]), torch.tensor([1, 1, 0, 0])
        )

        # Third batch: 3/3 correct
        training_metrics.update_train_batch(
            torch.tensor(0.8), torch.tensor([2, 1, 0]), torch.tensor([2, 1, 0])
        )

        assert training_metrics.train_total == 12  # 5 + 4 + 3
        assert training_metrics.train_correct == 9  # 4 + 2 + 3
        assert training_metrics.get_train_accuracy() == 75.0  # 9/12 * 100
        assert len(training_metrics.train_losses) == 3
        assert training_metrics.training_steps == 3

    def test_multiple_validation_batches_accuracy_accumulation(self, training_metrics):
        """Test accuracy calculation across multiple validation batches."""
        # First validation batch: 2/3 correct
        training_metrics.update_val_batch(torch.tensor(0.5), torch.tensor([1, 0, 1]), torch.tensor([1, 1, 1]))

        # Second validation batch: 1/2 correct
        training_metrics.update_val_batch(torch.tensor(0.7), torch.tensor([0, 1]), torch.tensor([0, 0]))

        assert training_metrics.val_total == 5  # 3 + 2
        assert training_metrics.val_correct == 3  # 2 + 1
        assert training_metrics.get_val_accuracy() == 60.0  # 3/5 * 100
        assert len(training_metrics.val_losses) == 2
        assert training_metrics.validation_steps == 2

    def test_reset_between_epochs_with_different_performance(self, training_metrics):
        """Test that metrics reset correctly between epochs with different performance characteristics."""
        # Simulate first epoch with poor performance
        training_metrics.update_train_batch(
            torch.tensor(2.0), torch.tensor([0, 1, 1, 0]), torch.tensor([1, 0, 1, 0])
        )
        first_epoch_accuracy = training_metrics.get_train_accuracy()
        first_epoch_losses = len(training_metrics.train_losses)

        # Reset for next epoch
        training_metrics.reset_epoch_metrics()

        # Simulate second epoch with better performance
        training_metrics.update_train_batch(
            torch.tensor(0.5), torch.tensor([1, 1, 1, 1, 1]), torch.tensor([1, 1, 1, 1, 1])
        )
        second_epoch_accuracy = training_metrics.get_train_accuracy()
        second_epoch_losses = len(training_metrics.train_losses)

        # Verify epoch isolation
        assert first_epoch_accuracy == 50.0  # 2/4 correct
        assert second_epoch_accuracy == 100.0  # 5/5 correct
        assert first_epoch_losses == 1
        assert second_epoch_losses == 1
        assert training_metrics.train_total == 5  # Only second epoch counts
        assert training_metrics.train_correct == 5  # Only second epoch counts

    def test_mixed_regression_classification_batches_in_single_epoch(self, training_metrics):
        """Test handling of mixed regression and classification batches within the same epoch."""
        # Regression batch (no predictions/targets)
        training_metrics.update_train_batch(torch.tensor(2.0))

        # Classification batch with partial accuracy
        training_metrics.update_train_batch(
            torch.tensor(1.0), torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1])
        )

        # Another regression batch
        training_metrics.update_train_batch(torch.tensor(1.5))

        # Classification batch with perfect accuracy
        training_metrics.update_train_batch(torch.tensor(0.8), torch.tensor([1, 0]), torch.tensor([1, 0]))

        # Verify mixed batch handling
        assert len(training_metrics.train_losses) == 4
        assert training_metrics.train_total == 5  # 3 + 2 from classification batches
        assert training_metrics.train_correct == 4  # 2 + 2 from classification batches
        assert training_metrics.get_train_accuracy() == 80.0  # 4/5 * 100
        assert training_metrics.training_steps == 4

    def test_edge_case_empty_tensors_in_classification(self, training_metrics):
        """Test behavior with empty tensors in classification updates."""
        loss = torch.tensor(1.0)
        empty_predictions = torch.tensor([])
        empty_targets = torch.tensor([])

        training_metrics.update_train_batch(loss, empty_predictions, empty_targets)

        assert training_metrics.train_total == 0
        assert training_metrics.train_correct == 0
        assert training_metrics.get_train_accuracy() == 0.0
        assert training_metrics.training_steps == 1

    def test_edge_case_single_sample_classification(self, training_metrics):
        """Test classification with single sample batches."""
        loss = torch.tensor(0.5)
        single_prediction = torch.tensor([1])
        single_target = torch.tensor([1])

        training_metrics.update_train_batch(loss, single_prediction, single_target)

        assert training_metrics.train_total == 1
        assert training_metrics.train_correct == 1
        assert training_metrics.get_train_accuracy() == 100.0

    def test_multiple_reset_calls_idempotency(self, training_metrics, sample_tensors):
        """Test that multiple calls to reset_epoch_metrics are idempotent."""
        # Add some data
        training_metrics.train_losses = [sample_tensors["loss_small"]]
        training_metrics.train_correct = 5
        training_metrics.train_total = 10
        training_metrics.training_steps = 3

        # Reset multiple times
        training_metrics.reset_epoch_metrics()
        training_metrics.reset_epoch_metrics()
        training_metrics.reset_epoch_metrics()

        # Verify state remains consistent
        assert training_metrics.train_losses == []
        assert training_metrics.train_correct == 0
        assert training_metrics.train_total == 0
        assert training_metrics.training_steps == 3  # Preserved

    def test_accuracy_calculation_precision(self, training_metrics):
        """Test accuracy calculation precision with edge cases."""
        # Test case that produces repeating decimal
        training_metrics.train_correct = 1
        training_metrics.train_total = 3

        accuracy = training_metrics.get_train_accuracy()
        expected = 100 * 1 / 3  # 33.33333333333333...

        assert accuracy == expected
        assert isinstance(accuracy, float)
