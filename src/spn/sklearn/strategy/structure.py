import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from typing import List

from spn.algorithms.LearningWrappers import learn_classifier, learn_parametric
from spn.structure.Base import Context, Node
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.Parametric import Parametric


class StructureLearningStrategy(ABC):
    """
    Abstract Base Class for structure learning strategies. Strategies have to implement :func:`learn_structure`.
"""

    def __init__(self):
        pass

    @abstractmethod
    def learn_structure(self, X: np.ndarray, y: np.ndarray) -> Node:
        """
        This method should implement the strategy of learning an SPN structure from the given data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target feature.

        Returns:
            Node: SPN root node of the learnt structure.
        """
        pass


class LearnClassifierParametric(StructureLearningStrategy):
    """Structure learner, that uses :func:`learn_classifier` with :func:`learn_parametric` internally."""

    def __init__(self, n_jobs=-1, parametric_types: List[Parametric] = None, **learn_parametric_kwargs):
        """
        Initialize the structure learner.

        Args:
            n_jobs (int): Number of jobs used for independence tests in :func:`learn_classifier`.
            parametric_types (List[Parametric]): List of parametric types for each feature in the input dataset.
            **learn_parametric_kwargs: Keyword arguments for :func:`learn_parametric`.
        """
        super().__init__()
        self.parametric_types = parametric_types
        self.learn_parametric_kwargs = learn_parametric_kwargs
        self.n_jobs = n_jobs

    def learn_structure(self, X: np.ndarray, y: np.ndarray) -> Node:
        # Merge X and y
        train_data = np.c_[X, y].astype(np.float32)

        # If no parametric types were given: Assume that all leafs are gaussian
        if self.parametric_types is None:
            parametric_types = [Gaussian] * X.shape[1] + [Categorical]
        else:
            parametric_types = self.parametric_types

        # Learn classifier
        spn = learn_classifier(
            train_data,
            ds_context=Context(parametric_types=parametric_types).add_domains(train_data),
            spn_learn_wrapper=partial(learn_parametric, **self.learn_parametric_kwargs),
            label_idx=X.shape[1],
            cpus=self.n_jobs,
        )

        return spn
