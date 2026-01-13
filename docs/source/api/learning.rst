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

Structures: HCLT
----------------

Hidden Chow–Liu Trees (HCLT) structure construction for binary and categorical data.

.. autofunction:: spflow.learn.hclt.learn_hclt_binary
.. autofunction:: spflow.learn.hclt.learn_hclt_categorical

Continuous Mixtures
-------------------

Continuous mixtures of tractable probabilistic models (RQMC integration + compilation).

.. autoclass:: spflow.learn.continuous_mixtures.LatentOptimizationConfig
   :members:
   :show-inheritance:

.. autofunction:: spflow.learn.continuous_mixtures.learn_continuous_mixture_factorized
.. autofunction:: spflow.learn.continuous_mixtures.learn_continuous_mixture_cltree

SOCS Builder
------------

Build a SOCS model from a compatible template circuit.

.. autofunction:: spflow.learn.build_socs.build_socs
