Random and Tensorized Sum-Product Networks (RAT-SPN)
====================================================

Random and Tensorized Sum-Product Networks (RAT-SPNs) provide a principled approach to building deep probabilistic models through randomized circuit construction. They combine interpretability with expressiveness through tensorized operations.

Reference
---------

RAT-SPNs are described in the NeurIPS 2020 paper:

- `Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning <https://proceedings.neurips.cc/paper/2020/hash/791ad60a80e6f66318e88863c0a51c4a-Abstract.html>`_

Overview
--------

RAT-SPNs consist of alternating sum (region) and product (partition) layers that recursively partition the input space. The random construction prevents overfitting while maintaining tractable exact inference.

Key features:
~~~~~~~~~~~~~

- **Randomized structure**: Region and partition layers are constructed using random permutations and splits.
- **Tensorized evaluation**: Operations are mapped to efficient tensor contractions.
- **Scalable training**: Supports training via EM or Gradient Descent.

Implementation
--------------

The RAT-SPN implementation in SPFlow provides a high-level :class:`spflow.zoo.rat.RatSPN` module that automates the construction of the circuit based on architectural hyperparameters.

.. autoclass:: spflow.zoo.rat.RatSPN
   :members:
   :show-inheritance:
