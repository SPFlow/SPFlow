"""Abstract base class for classification modules."""

from abc import ABC, abstractmethod

import torch


class Classifier(ABC):
    """Abstract base class for modules that support classification.

    Provides a standard interface for models that can predict class labels
    and class probabilities from input data.
    """

    @abstractmethod
    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for input data.

        Args:
            data: Input data tensor.

        Returns:
            Class probability predictions. Each row corresponds to a data point,
            and each column corresponds to a class.
        """
        pass

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Predict class labels for input data.

        Args:
            data: Input data tensor.

        Returns:
            Predicted class labels.
        """
        return torch.argmax(self.predict_proba(data), dim=1)
