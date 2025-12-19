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

SplitMode
---------

Factory class for creating split configurations.

.. autoclass:: spflow.modules.ops.split.SplitMode
   :members: consecutive, interleaved, by_index, create

SplitConsecutive
-----------

Splits features into consecutive halves (or n parts).

.. autoclass:: spflow.modules.ops.SplitConsecutive

SplitInterleaved
--------------

Splits features in alternating fashion.

.. autoclass:: spflow.modules.ops.SplitInterleaved

SplitByIndex
------------

Splits features according to user-specified indices.

.. autoclass:: spflow.modules.ops.SplitByIndex
