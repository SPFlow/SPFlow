Utilities
=========

Helper functions and utilities for model visualization, I/O, and analysis.

Visualization
-------------

Visualize probabilistic circuit structures as graphs.

.. autofunction:: spflow.utils.visualization.visualize

Model I/O
---------

Save and load models from disk.

.. autofunction:: spflow.utils.model_manager.save_model

.. autofunction:: spflow.utils.model_manager.load_model


Cache
-----

Utilities for caching intermediate computations to speed up inference.

.. autoclass:: spflow.utils.cache.Cache

RQMC
----

Randomized quasi-Monte Carlo helpers used for continuous mixtures.

.. note::

   This feature is experimental and located in :mod:`spflow.exp.cms`.

.. autoclass:: spflow.exp.cms.RqmcPoints
   :members:
   :show-inheritance:

.. autofunction:: spflow.exp.cms.rqmc_sobol_normal

Method Replacement
------------------

Temporarily replace module methods for testing or experimentation.

.. autofunction:: spflow.utils.replace.replace
