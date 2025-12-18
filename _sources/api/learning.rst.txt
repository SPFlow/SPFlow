Learning and Training
=====================

Structure and parameter learning algorithms for probabilistic circuits.

Structure Learning
------------------

Automatic structure learning using the LearnSPN algorithm based on Randomized Dependence Coefficients (RDC).

.. autofunction:: spflow.learn.learn_spn.learn_spn

Parameter Learning: EM
----------------------

Expectation-Maximization algorithm for parameter optimization.

.. autofunction:: spflow.learn.expectation_maximization.expectation_maximization

Parameter Learning: Gradient Descent
-------------------------------------

Gradient descent-based parameter learning using PyTorch optimizers.

.. autofunction:: spflow.learn.gradient_descent.train_gradient_descent
