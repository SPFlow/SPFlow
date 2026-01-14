Learning and Training
=====================

Structure and parameter learning algorithms for probabilistic circuits.

Structure Learning
------------------

Automatic structure learning using the LearnSPN algorithm based on Randomized Dependence Coefficients (RDC).

.. autofunction:: spflow.learn.learn_spn.learn_spn

Automatic structure learning using the Prometheus algorithm for learning DAG-structured SPNs (with optional subtree sharing).

.. autofunction:: spflow.learn.prometheus.learn_prometheus

Parameter Learning: EM
----------------------

Expectation-Maximization algorithm for parameter optimization.

.. autofunction:: spflow.learn.expectation_maximization.expectation_maximization

Parameter Learning: Gradient Descent
-------------------------------------

Gradient descent-based parameter learning using PyTorch optimizers.

.. autofunction:: spflow.learn.gradient_descent.train_gradient_descent

SOCS Builder
------------

Build a SOCS model from a compatible template circuit.

.. autofunction:: spflow.learn.build_socs.build_socs
