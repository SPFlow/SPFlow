Leaf Modules
============

Probability distributions at the leaves of the probabilistic circuit. See :doc:`base_modules` for the abstract LeafModule base class.

Continuous Distributions
-------------------------

Normal
^^^^^^

Gaussian distribution with learnable mean and standard deviation.

.. autoclass:: spflow.modules.leaves.Normal

LogNormal
^^^^^^^^^

Log-normal distribution (exponential of a normal distribution).

.. autoclass:: spflow.modules.leaves.LogNormal

Exponential
^^^^^^^^^^^

Exponential distribution with learnable rate parameter.

.. autoclass:: spflow.modules.leaves.Exponential

Gamma
^^^^^

Gamma distribution with learnable alpha and beta parameters.

.. autoclass:: spflow.modules.leaves.Gamma

Uniform
^^^^^^^

Uniform distribution over a bounded interval.

.. autoclass:: spflow.modules.leaves.Uniform

Discrete Distributions
----------------------

Categorical
^^^^^^^^^^^

Categorical distribution over K categories with learnable probabilities.

.. autoclass:: spflow.modules.leaves.Categorical

Bernoulli
^^^^^^^^^

Bernoulli distribution (binary outcomes) with learnable probability.

.. autoclass:: spflow.modules.leaves.Bernoulli

Binomial
^^^^^^^^

Binomial distribution with learnable probability and fixed number of trials.

.. autoclass:: spflow.modules.leaves.Binomial

Poisson
^^^^^^^

Poisson distribution with learnable rate parameter.

.. autoclass:: spflow.modules.leaves.Poisson

Geometric
^^^^^^^^^

Geometric distribution with learnable success probability.

.. autoclass:: spflow.modules.leaves.Geometric

NegativeBinomial
^^^^^^^^^^^^^^^^

Negative binomial distribution with learnable parameters.

.. autoclass:: spflow.modules.leaves.NegativeBinomial

Hypergeometric
^^^^^^^^^^^^^^

Hypergeometric distribution.

.. autoclass:: spflow.modules.leaves.Hypergeometric
