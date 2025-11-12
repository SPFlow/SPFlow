API Reference
=============

Complete API documentation for all SPFlow modules, classes, and functions.

Core API
--------

.. toctree::
   :maxdepth: 2

   modules
   leaf
   learning

Dispatched Functions
--------------------

SPFlow uses polymorphic dispatch for key operations. These functions work across all module types:

.. currentmodule:: spflow

.. autosummary::
   :nosignatures:

   log_likelihood
   sample
   sample_with_evidence
   em
   maximum_likelihood_estimation
   marginalize

.. autofunction:: log_likelihood
.. autofunction:: sample
.. autofunction:: sample_with_evidence
.. autofunction:: em
.. autofunction:: maximum_likelihood_estimation
.. autofunction:: marginalize

Supporting Components
---------------------

.. toctree::
   :maxdepth: 2

   distributions
   meta
   utils
   exceptions

Quick Links
-----------

**Common Classes:**

- :class:`spflow.modules.Module` - Base module class
- :class:`spflow.modules.Sum` - Sum nodes (mixtures)
- :class:`spflow.modules.Product` - Product nodes (factorization)
- :class:`spflow.modules.leaf.Normal` - Gaussian leaf distribution
- :class:`spflow.modules.leaf.Categorical` - Categorical leaf distribution
- :class:`spflow.modules.rat.RatSPN` - RAT-SPN architecture
- :class:`spflow.meta.Scope` - Feature scope tracking

**Learning Functions:**

- :func:`spflow.learn.learn_spn` - Structure learning (LearnSPN)
- :func:`spflow.learn.train_gradient_descent` - Gradient-based parameter learning
- :func:`spflow.learn.expectation_maximization` - EM-based parameter learning

**Utilities:**

- :func:`spflow.utils.visualization.visualize_module` - Visualize model structure
- :func:`spflow.utils.model_manager.save_model` - Save trained models
- :func:`spflow.utils.model_manager.load_model` - Load saved models

Index
-----

* :ref:`genindex`
* :ref:`modindex`
