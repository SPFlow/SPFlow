import numpy as np
from abc import ABC, abstractmethod

from spn.algorithms.MPE import mpe
from spn.structure.Base import Node


class PredictionStrategy(ABC):
    """
    A strategy that defines how predictions are made on new data, given a SPN.
    """

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, spn: Node, X: np.ndarray) -> np.ndarray:
        """
        Predict the target feature from the given input features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Target feature prediction for each instance in X.
        """
        pass


class MpePredictionStrategy(PredictionStrategy):
    """
    Prediction strategy that uses :func:`mpe` for the target features.
    """

    def __init__(self):
        super().__init__()

    def predict(self, spn: Node, X: np.ndarray) -> np.ndarray:
        # Get number of test instances
        n_test = X.shape[0]

        # Create empty y_hat vector
        y_empty = np.full((n_test, 1), fill_value=np.nan)

        # Merge test data and empty y_hat vector
        data = np.c_[X, y_empty]

        # Obtain predictions with mpe
        data_filled = mpe(spn, data)

        # Select  predictions
        y_pred = data_filled[:, -1]
        return y_pred
