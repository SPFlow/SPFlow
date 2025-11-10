Leaf Distributions
==================

Leaf nodes are probability distributions over input features. SPFlow supports both unconditional and conditional distributions.

Base Classes
------------

Leaf Module
^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.leaf_module

.. autoclass:: LeafModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Conditional Leaf Module
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.cond_leaf_module

.. autoclass:: CondLeafModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Continuous Distributions
-------------------------

Normal (Gaussian)
^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.normal

.. autoclass:: Normal
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Log-Normal
^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.log_normal

.. autoclass:: LogNormal
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Gamma
^^^^^

.. currentmodule:: spflow.modules.leaf.gamma

.. autoclass:: Gamma
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Exponential
^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.exponential

.. autoclass:: Exponential
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Uniform
^^^^^^^

.. currentmodule:: spflow.modules.leaf.uniform

.. autoclass:: Uniform
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Discrete Distributions
----------------------

Bernoulli
^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.bernoulli

.. autoclass:: Bernoulli
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Binomial
^^^^^^^^

.. currentmodule:: spflow.modules.leaf.binomial

.. autoclass:: Binomial
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Categorical
^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.categorical

.. autoclass:: Categorical
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Geometric
^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.geometric

.. autoclass:: Geometric
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Poisson
^^^^^^^

.. currentmodule:: spflow.modules.leaf.poisson

.. autoclass:: Poisson
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Hypergeometric
^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.hypergeometric

.. autoclass:: Hypergeometric
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Negative Binomial
^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.negative_binomial

.. autoclass:: NegativeBinomial
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Conditional Distributions
--------------------------

Conditional distributions have parameters that depend on parent node values.

.. note::
   Currently, only a subset of conditional distributions are implemented. Additional conditional distributions will be added in future releases.

Conditional Normal
^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.cond_normal

.. autoclass:: CondNormal
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Conditional Bernoulli
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.cond_bernoulli

.. autoclass:: CondBernoulli
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Conditional Binomial
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.cond_binomial

.. autoclass:: CondBinomial
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize

Conditional Categorical
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.leaf.cond_categorical

.. autoclass:: CondCategorical
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.maximum_likelihood_estimation
   spflow.marginalize
