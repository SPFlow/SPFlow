Convolutional Modules
=====================

Convolutional layers for modeling spatial structure in image data with probabilistic circuits.

ConvPc
------

High-level convolutional probabilistic circuit architecture that stacks alternating
ProdConv and SumConv layers on top of a leaf distribution. Similar to RAT-SPN but
designed specifically for image data with spatial structure.

.. autoclass:: spflow.modules.conv.ConvPc

SumConv
-------

Convolutional sum layer that applies learned weighted sums over input channels
within spatial patches. Enables mixture modeling with spatial structure.

.. autoclass:: spflow.modules.conv.SumConv

ProdConv
--------

Convolutional product layer that computes products over spatial patches,
reducing spatial dimensions by the kernel size factor. Aggregates scopes
within patches while maintaining proper probabilistic semantics.

.. autoclass:: spflow.modules.conv.ProdConv
