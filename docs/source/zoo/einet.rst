Einsum Networks (Einet)
=======================

Einsum Networks (Einets) are a scalable class of probabilistic circuits that use Einstein summation notation (einsum) to implement efficient sum-product operations in parallel.

Reference
---------

Einets are described in the ICML 2020 paper:

- `Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits <https://arxiv.org/abs/2004.06231>`_

Overview
--------

Einet provides a scalable architecture for Sum-Product Networks using ``EinsumLayer`` or ``LinsumLayer`` for efficient batched computations. These layers combine product and sum operations into single efficient einsum operations.

Key Characteristics:
~~~~~~~~~~~~~~~~~~~~

- **Efficient batched computations**: Leverage PyTorch's optimized ``einsum`` implementation.
- **Scalable deep architecture**: Supports deep stacks of einsum/linsum layers.
- **Fast inference and sampling**: Optimized for high-throughput probabilistic modeling.

Implementation
--------------

The Einet implementation in SPFlow provides a high-level :class:`spflow.zoo.einet.Einet` module.

.. autoclass:: spflow.zoo.einet.Einet
   :members:
   :show-inheritance:

Layers
------

.. autoclass:: spflow.modules.einsum.EinsumLayer
   :members:
   :show-inheritance:

.. autoclass:: spflow.modules.einsum.LinsumLayer
   :members:
   :show-inheritance:
