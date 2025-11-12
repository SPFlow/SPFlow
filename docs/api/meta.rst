Metadata System
===============

The metadata system in SPFlow manages scopes, feature contexts, and dispatch contexts for advanced functionality.

Data Management
---------------

Scope
^^^^^

The ``Scope`` class tracks which features (variables) a module operates on.

.. currentmodule:: spflow.meta.data.scope

.. autoclass:: Scope
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __contains__, __len__, __iter__, __or__, __and__, __eq__

Feature Context
^^^^^^^^^^^^^^^

The ``FeatureContext`` class stores metadata about features (continuous vs discrete, etc.).

.. currentmodule:: spflow.meta.data.feature_context

.. autoclass:: FeatureContext
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Feature Types
^^^^^^^^^^^^^

Enumeration of supported feature types.

.. currentmodule:: spflow.meta.data.feature_types

.. autoclass:: FeatureTypes
   :members:
   :undoc-members:
   :show-inheritance:

Meta Type
^^^^^^^^^

Base class for metadata types.

.. currentmodule:: spflow.meta.data.meta_type

.. autoclass:: MetaType
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Dispatch System
---------------

Dispatch
^^^^^^^^

Core dispatch mechanism for polymorphic function calls.

.. currentmodule:: spflow.meta.dispatch.dispatch

.. autofunction:: dispatch

.. autoclass:: Dispatcher
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Dispatch Context
^^^^^^^^^^^^^^^^

Context object passed through dispatched function calls.

.. currentmodule:: spflow.meta.dispatch.dispatch_context

.. autoclass:: DispatchContext
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Sampling Context
^^^^^^^^^^^^^^^^

Context for sampling operations.

.. currentmodule:: spflow.meta.dispatch.sampling_context

.. autoclass:: SamplingContext
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Memoization
^^^^^^^^^^^

Memoization support for dispatched functions.

.. currentmodule:: spflow.meta.dispatch.memoize

.. autofunction:: memoize

.. autoclass:: MemoizationCache
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Substitutable
^^^^^^^^^^^^^

Support for substitution in dispatched calls.

.. currentmodule:: spflow.meta.dispatch.substitutable

.. autoclass:: Substitutable
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Working with Scopes
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow.meta import Scope

   # Create scopes
   scope1 = Scope([0, 1, 2])
   scope2 = Scope([2, 3, 4])

   # Scope operations
   union = scope1 | scope2  # Scope([0, 1, 2, 3, 4])
   intersection = scope1 & scope2  # Scope([2])

   # Check membership
   assert 1 in scope1
   assert 5 not in scope1

   # Iterate over features
   for feature in scope1:
       print(feature)

Using Feature Context
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow.meta import FeatureContext, FeatureTypes

   # Define feature types
   ctx = FeatureContext([
       FeatureTypes.CONTINUOUS,  # Feature 0
       FeatureTypes.DISCRETE,    # Feature 1
       FeatureTypes.CONTINUOUS,  # Feature 2
   ])

   # Query feature information
   print(ctx.is_continuous(0))  # True
   print(ctx.is_discrete(1))    # True

Sampling Context
^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from spflow.meta import SamplingContext

   n_samples = 10
   n_features = 4

   # Set up sampling context
   channel_index = torch.zeros(n_samples, n_features, dtype=torch.int64)
   mask = torch.ones(n_samples, n_features, dtype=torch.bool)

   ctx = SamplingContext(
       channel_index=channel_index,
       mask=mask
   )

   # Use in sampling
   from spflow import sample
   samples = sample(model, evidence, sampling_ctx=ctx)

Dispatch Context
^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow.meta import DispatchContext

   ctx = DispatchContext(
       memoize=True,  # Enable memoization
       gradient_tracking=True  # Track gradients
   )

   # Use with dispatched functions
   from spflow import log_likelihood
   ll = log_likelihood(model, data, ctx=ctx)
