Base Classes
============

Core infrastructure classes that form the foundation of SPFlow's module system.

Module
------

The abstract base class for all SPFlow modules. Every probabilistic circuit component inherits from this class.

.. autoclass:: spflow.modules.module.Module
   :members:
   :no-index:


LeafModule
----------

Abstract base class for all probability distribution implementations at the leaves of the circuit.

.. autoclass:: spflow.modules.leaves.leaf.LeafModule

