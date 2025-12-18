Einsum Modules
==============

Efficient sum-product operations using Einstein summation notation, as described in the
`EinsumNetworks paper <https://arxiv.org/abs/2004.06231>`_. These layers combine product
and sum operations into single efficient einsum operations.

Einet
-----

High-level architecture for building Einsum Networks, a scalable deep probabilistic
model using EinsumLayer or LinsumLayer for efficient batched computations.

**Key parameters:**

- ``num_classes``: Number of root sum nodes (for classification)
- ``num_sums``: Number of sum nodes per intermediate layer
- ``num_leaves``: Number of leaf distribution components
- ``depth``: Number of einsum layers (determines feature grouping: 2^depth features)
- ``num_repetitions``: Number of parallel circuit repetitions
- ``layer_type``: ``"einsum"`` (cross-product) or ``"linsum"`` (linear combination)
- ``structure``: ``"top-down"`` or ``"bottom-up"`` construction mode

**Reference:** Peharz, R., et al. (2020). "Einsum Networks: Fast and Scalable Learning
of Tractable Probabilistic Circuits." ICML 2020.

.. autoclass:: spflow.modules.einsum.Einet
   :members:
   :undoc-members:
   :show-inheritance:

EinsumLayer
-----------

Combines product and sum operations using a cross-product over input channels.
Takes pairs of adjacent features as left/right children, computes their cross-product
over channels (I × J combinations), and sums with learned weights using the
LogEinsumExp trick for numerical stability.

**Key characteristics:**

- Weight shape: ``(features, out_channels, repetitions, left_channels, right_channels)``
- Computes cross-product: I × J input channel combinations
- Uses LogEinsumExp for numerical stability in log-space

.. autoclass:: spflow.modules.einsum.EinsumLayer
   :members:
   :undoc-members:
   :show-inheritance:

LinsumLayer
-----------

Linear sum-product layer with a simpler linear combination over channels.
Unlike EinsumLayer which computes a cross-product (I × J), LinsumLayer pairs
left/right features, adds them (product in log-space), then sums over input
channels with learned weights.

**Key characteristics:**

- Weight shape: ``(features, out_channels, repetitions, in_channels)``
- Linear combination: requires left and right inputs to have matching channel counts
- Fewer parameters than EinsumLayer: O(C) vs O(C²)

.. autoclass:: spflow.modules.einsum.LinsumLayer
   :members:
   :undoc-members:
   :show-inheritance:

Comparison
----------

+----------------+-------------------------+--------------------+-------------------------+
| Layer          | Weight Shape            | Channel Operation  | Parameter Count         |
+================+=========================+====================+=========================+
| EinsumLayer    | (D, O, R, I, J)         | Cross-product I×J  | O(D · O · R · I · J)    |
+----------------+-------------------------+--------------------+-------------------------+
| LinsumLayer    | (D, O, R, C)            | Linear sum         | O(D · O · R · C)        |
+----------------+-------------------------+--------------------+-------------------------+

Use **EinsumLayer** when you need maximum expressiveness with different left/right channel counts.
Use **LinsumLayer** when you want fewer parameters and have matching channel counts.
