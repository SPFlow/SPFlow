Operations
===========

Structural operations for combining, splitting, and factorizing features in probabilistic circuits.

Cat
---

Concatenates multiple modules along the feature or channel dimension.

.. autoclass:: spflow.modules.ops.Cat

Split
-----

Abstract base class for feature splitting operations.

.. autoclass:: spflow.modules.ops.split.Split

SplitHalves
-----------

Splits features into consecutive halves (or n parts).

.. autoclass:: spflow.modules.ops.SplitHalves

SplitAlternate
--------------

Splits features in alternating fashion.

.. autoclass:: spflow.modules.ops.SplitAlternate

