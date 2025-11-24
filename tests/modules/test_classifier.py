"""Test cases for the Classifier abstract base class."""

import pytest
import torch

from spflow.interfaces.classifier import Classifier


class ConcreteClassifier(Classifier):
    """Concrete implementation of Classifier for testing."""

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        """Return dummy probabilities.

        Args:
            data: Input data tensor of shape (batch_size, num_features).

        Returns:
            Class probabilities of shape (batch_size, num_classes).
        """
        batch_size = data.shape[0]
        num_classes = 3
        # Return dummy normalized probabilities
        probs = torch.ones(batch_size, num_classes)
        return probs / num_classes

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Return dummy class predictions.

        Args:
            data: Input data tensor of shape (batch_size, num_features).

        Returns:
            Predicted class labels of shape (batch_size,).
        """
        batch_size = data.shape[0]
        # Return dummy class predictions (0, 1, or 2)
        return torch.zeros(batch_size, dtype=torch.long)


class TestClassifier:
    """Test cases for the Classifier abstract base class."""

    def test_classifier_is_abstract(self):
        """Test that Classifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Classifier()

    def test_concrete_classifier_instantiation(self):
        """Test that concrete implementation can be instantiated."""
        classifier = ConcreteClassifier()
        assert isinstance(classifier, Classifier)

    def test_predict_proba_method_exists(self):
        """Test that concrete classifier has predict_proba method."""
        classifier = ConcreteClassifier()
        assert hasattr(classifier, "predict_proba")
        assert callable(getattr(classifier, "predict_proba"))

    def test_predict_method_exists(self):
        """Test that concrete classifier has predict method."""
        classifier = ConcreteClassifier()
        assert hasattr(classifier, "predict")
        assert callable(getattr(classifier, "predict"))

    def test_predict_proba_output_shape(self):
        """Test that predict_proba returns correct output shape."""
        classifier = ConcreteClassifier()
        batch_size = 5
        num_features = 10
        data = torch.randn(batch_size, num_features)

        probs = classifier.predict_proba(data)

        assert isinstance(probs, torch.Tensor)
        assert probs.shape[0] == batch_size

    def test_predict_output_shape(self):
        """Test that predict returns correct output shape."""
        classifier = ConcreteClassifier()
        batch_size = 5
        num_features = 10
        data = torch.randn(batch_size, num_features)

        predictions = classifier.predict(data)

        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape[0] == batch_size

    def test_predict_proba_with_single_sample(self):
        """Test predict_proba with a single sample."""
        classifier = ConcreteClassifier()
        data = torch.randn(1, 10)

        probs = classifier.predict_proba(data)

        assert probs.shape[0] == 1

    def test_predict_with_single_sample(self):
        """Test predict with a single sample."""
        classifier = ConcreteClassifier()
        data = torch.randn(1, 10)

        predictions = classifier.predict(data)

        assert predictions.shape[0] == 1

    def test_predict_proba_with_large_batch(self):
        """Test predict_proba with large batch size."""
        classifier = ConcreteClassifier()
        batch_size = 1000
        num_features = 10
        data = torch.randn(batch_size, num_features)

        probs = classifier.predict_proba(data)

        assert probs.shape[0] == batch_size

    def test_predict_with_large_batch(self):
        """Test predict with large batch size."""
        classifier = ConcreteClassifier()
        batch_size = 1000
        num_features = 10
        data = torch.randn(batch_size, num_features)

        predictions = classifier.predict(data)

        assert predictions.shape[0] == batch_size

    def test_predict_proba_returns_tensor(self):
        """Test that predict_proba returns a torch.Tensor."""
        classifier = ConcreteClassifier()
        data = torch.randn(5, 10)

        result = classifier.predict_proba(data)

        assert isinstance(result, torch.Tensor)

    def test_predict_returns_tensor(self):
        """Test that predict returns a torch.Tensor."""
        classifier = ConcreteClassifier()
        data = torch.randn(5, 10)

        result = classifier.predict(data)

        assert isinstance(result, torch.Tensor)

    def test_predict_proba_with_different_feature_dimensions(self):
        """Test predict_proba with different feature dimensions."""
        classifier = ConcreteClassifier()

        for num_features in [1, 5, 10, 100]:
            data = torch.randn(3, num_features)
            probs = classifier.predict_proba(data)
            assert probs.shape[0] == 3

    def test_predict_with_different_feature_dimensions(self):
        """Test predict with different feature dimensions."""
        classifier = ConcreteClassifier()

        for num_features in [1, 5, 10, 100]:
            data = torch.randn(3, num_features)
            predictions = classifier.predict(data)
            assert predictions.shape[0] == 3

    def test_abstract_method_not_implemented_in_subclass(self):
        """Test that subclass without implementing abstract methods raises TypeError."""

        class IncompleteClassifier(Classifier):
            """Incomplete implementation missing predict_proba."""

            def predict(self, data: torch.Tensor) -> torch.Tensor:
                return torch.zeros(data.shape[0])

        with pytest.raises(TypeError):
            IncompleteClassifier()

    def test_predict_proba_abstract_enforcement(self):
        """Test that predict_proba must be implemented."""

        class NoPredict(Classifier):
            """Missing predict_proba implementation."""

            def predict(self, data: torch.Tensor) -> torch.Tensor:
                return torch.zeros(data.shape[0])

        with pytest.raises(TypeError):
            NoPredict()

    def test_predict_default_implementation(self):
        """Test that predict has a default implementation based on predict_proba."""

        class MinimalClassifier(Classifier):
            """Minimal implementation with only predict_proba."""

            def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
                batch_size = data.shape[0]
                # Return dummy probabilities with shape (batch_size, num_classes)
                return torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]])

        classifier = MinimalClassifier()
        data = torch.randn(3, 10)
        predictions = classifier.predict(data)

        # Should use argmax of predict_proba
        # predict_proba returns [[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]]
        # argmax gives [1, 0, 0]
        assert predictions.shape[0] == 3
        assert torch.allclose(predictions, torch.tensor([1, 0, 0]))
