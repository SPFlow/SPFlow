import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from spn.gpu.TensorFlow import optimize_tf
from spn.structure.Base import Node


class OptimizationStrategy(ABC):
    """
    Strategy to optimize the weights of a given SPN.
    """

    def __init__(self):
        pass

    @abstractmethod
    def optimize(self, spn: Node, X: np.ndarray, y: np.ndarray) -> Node:
        """
        Optimize the weights of a given SPN.

        Args:
            spn (Node): Root node of the SPN.
            X (np.ndarray): Input features.
            y (np.ndarray): Target variable.

        Returns:
            Node: Root node of the optimized SPN.
        """
        pass


class NoOpOptimization(OptimizationStrategy):
    """
    No operation optimization strategy that does not optimize the SPN but simply forwards the SPN.
    """

    def __init__(self):
        super().__init__()

    def optimize(self, spn: Node, X: np.ndarray, y: np.ndarray) -> Node:
        return spn


class TensorFlowOptimization(OptimizationStrategy):
    """
    TensorFlow optimization Strategy that compiles a given SPN into a static tensorflow graph and uses gradient
    descent to optimize the weights regarding the log likelihood of the given data.
    """

    def __init__(self, batch_size: int, n_epochs: int, optimizer=tf.train.AdamOptimizer(learning_rate=0.01)):
        """
        Initialze the TensorFlowOptimizationStrategy.

        After :func:`optimize` is called, the :class:`TensorFlowOptimization` object stores the optimization
        loss for each epoch in the :attr:`loss` list attribute.

        Args:
           batch_size (int): Tensorflow batch size.
           n_epochs (int): Number of training epochs.
           optimizer: Tensorflow optimizer that is used to apply the gradient descent updates on the SPN weights.
        """
        super().__init__()
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.loss = []

    def optimize(self, spn: Node, X: np.ndarray, y: np.ndarray) -> Node:
        train_data = np.c_[X, y].astype(np.float32)
        spn, self.loss = optimize_tf(
            spn=spn,
            data=train_data,
            optimizer=self.optimizer,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            return_loss=True,
        )

        return spn


def classification_categorical_to_tf_graph(
    node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32
):
    """
    Fix categorical to tf graph for classification problem.

    For a binary class label, there will be two categorical leaf nodes in the SPN. One which one-hot encodes the first
    class as [0, 1] and one that encodes the second class as [1, 0].

    Since the tf optimizes the log likelihood, these one-hot represented probabilities will be projected into logspace
    which results in log([1,0])=[0, -inf] and therefore NaNs in further computations.

    Therefore, this custom method adds a small epsilon, such that the zero probability value in the one-hot vector will
    not degrade to negative infinity.
    """
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        p = np.array(node.p, dtype=dtype)

        # Epsilon to make sure there are no zero values
        eps = 1e-10
        p += eps

        # Renormalize such that the sum over all probabilities is one
        p /= np.sum(p)

        assert np.all(p > 0), "Probabilities in the class leaf nodes have to be greater than zero but were %s" % p

        softmaxInverse = np.log(p / np.max(p)).astype(dtype)
        probs = tf.nn.softmax(tf.constant(softmaxInverse))
        variable_dict[node] = probs
        if log_space:
            return tf.distributions.Categorical(probs=probs).log_prob(data_placeholder[:, node.scope[0]])

        return tf.distributions.Categorical(probs=probs).prob(data_placeholder[:, node.scope[0]])
