Convolutional Probabilistic Circuits (ConvPc)
=============================================

Convolutional Probabilistic Circuits (ConvPc) are a multi-layer architecture that stacks alternating SumConv and ProdConv layers on top of a leaf distribution, designed specifically for data with spatial structure like images.

Reference
---------

Convolutional architectures for Sum-Product Networks are inspired by:

- `Convolutional Sum-Product Networks <https://arxiv.org/abs/1902.04687>`_ (Butko & Zhang, 2019)

Overview
--------

ConvPc architectures progressively reduce spatial dimensions while learning mixture weights at each level, mirroring the hierarchical structure of Convolutional Neural Networks (CNNs) while maintaining exact tractability.

Key characteristics:
~~~~~~~~~~~~~~~~~~~~

- **Spatial awareness**: Uses local kernels to capture spatial correlations.
- **Weight sharing**: (Optionally) shares weights across spatial locations for efficiency.
- **Hierarchical composition**: Recursively combines local distributions into global ones.

Implementation
--------------

The :class:`spflow.zoo.conv.ConvPc` module automates the construction of these circuits for image modeling.

.. autoclass:: spflow.zoo.conv.ConvPc
   :members:
   :show-inheritance:
