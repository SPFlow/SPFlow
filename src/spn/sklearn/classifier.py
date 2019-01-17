import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array, check_is_fitted

from spn.sklearn.strategy.optimization import NoOpOptimization, OptimizationStrategy
from spn.sklearn.strategy.prediction import MpePredictionStrategy, PredictionStrategy
from spn.sklearn.strategy.structure import LearnClassifierParametric, StructureLearningStrategy

logger = logging.getLogger(__name__)


class SPNClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that wraps SPN functionality into the sklearn estimator interface to expose SPNs to sklearn methods
    such as :func:`sklearn.model_selection.cross_val_score` or :class:`sklearn.model_selection.GridSearchCV`.

    The classifier mainly acts as a pipeline of three main abstract parts:

    1) Structure learning: Here, the SPN structure is derived from a given dataset. A strategy for structure learning
        can be chosen from :mod:`spn.sklearn.strategy.structure`.

    2) Weight optimization: The second step consists of a strategy defining how the weights in the SPN are going to be
        further optimized. This can e.g. be done with gradient descent optimization from
        :class:`spn.sklearn.strategy.optimization.TensorFlowOptimizationStrategy`. Optimization strategies can be found
        in :mod:`spn.sklearn.strategy.optimization`.

    3) Prediction strategy: The final part of the pipeline defines how to generate predictions for input data.
        Strategies for predictions can be found in :mod:`spn.sklearn.strategy.prediction`.


    Example:

        .. code:: python

            from sklearn.datasets import load_iris
            from sklearn.model_selection import cross_val_score

            from spn.gpu.TensorFlow import add_node_to_tf_graph
            from spn.sklearn.classifier import SPNClassifier
            from spn.sklearn.strategy.structure import LearnClassifierParametric
            from spn.sklearn.strategy.optimization import classification_categorical_to_tf_graph
            from spn.structure.leaves.parametric.Parametric import Categorical

            add_node_to_tf_graph(Categorical, classification_categorical_to_tf_graph)
            X, y = load_iris(return_X_y=True)
            sl = LearnClassifierParametric(min_instances_slice=25)
            clf = SPNClassifier(structure_learner=sl)
            scores = cross_val_score(clf, X, y, cv=10, n_jobs=4)
            print("Scores: ", scores)

            # Scores:  [1.         0.93333333 1.         0.93333333 0.93333333 0.93333333
            # 0.93333333 1.         1.         1.        ]

    """

    def __init__(
        self,
        structure_learner: StructureLearningStrategy = LearnClassifierParametric(),
        optimizer: OptimizationStrategy = NoOpOptimization(),
        predictor: PredictionStrategy = MpePredictionStrategy(),
    ):
        """
        Initialize a SPNClassifier.

        Args:
            structure_learner (StructureLearningStrategy): Strategy for learning the SPN structure from data.
            optimizer (OptimizationStrategy): Strategy to optimize the SPN weights.
            predictor (PredictionStrategy): Strategy to generate predictions from new data.
        """
        self.structure_learner = structure_learner
        self.optimizer = optimizer
        self.predictor = predictor

    def fit(self, X, y):
        """
        Fit the estimator object.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training target variable.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True)

        # Apply learning strategy
        self._spn = self.structure_learner.learn_structure(X, y)

        # Apply weight optimization
        self._spn = self.optimizer.optimize(self._spn, X, y)

        # Store X and y to mark estimator as fitted
        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X: np.ndarray):
        """
        Use the SPN to generate predictions for new data.

        Args:
            X (np.ndarray): Input features.
        """
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X)

        y_pred = self.predictor.predict(self._spn, X)
        return y_pred

    def get_params(self, deep=True):
        """Method to make SPNClassifier usable in sklearn procedures such as cross_val_score etc."""
        return {"structure_learner": self.structure_learner, "optimizer": self.optimizer, "predictor": self.predictor}

    def set_params(self, **parameters):
        """Method to make SPNClassifier usable in sklearn procedures such as cross_val_score etc."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
