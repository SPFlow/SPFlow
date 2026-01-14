Hidden Chow-Liu Trees (HCLT)
============================

Hidden Chow-Liu Trees (HCLTs) are latent-variable models where the structure is derived from a Chow-Liu tree over the observed variables, and hidden states are modeled via the channel dimension.

Reference
---------

HCLTs and their learning algorithms are discussed in:

- `Learning Hidden Chow-Liu Trees <https://arxiv.org/abs/2106.10332>`_ (Liu & Van den Broeck, 2021)
- `Probabilistic Circuits: A Unifying Framework for Tractable Probabilistic Models <https://starai.cs.ucla.edu/papers/ChoiProbCirc20.pdf>`_ (Choi et al., 2020)

Overview
--------

HCLTs represent a powerful compromise between the simplicity of tree-structured models and the expressiveness of deep circuits. By introducing latent variables at each node of a Chow-Liu tree, they can capture complex dependencies while remaining extremely efficient to learn and evaluate.

Key features:
~~~~~~~~~~~~~

- **Structure learning**: Uses the Chow-Liu algorithm to find the optimal tree structure.
- **Top-k Mixtures**: Supports building mixtures over multiple high-scoring trees for increased robustness.
- **Latent states**: Each observed variable is associated with a hidden category that mediates its dependencies.

Implementation
--------------

SPFlow provides automated learners for binary and categorical HCLTs.

Binary HCLT
-----------

.. autofunction:: spflow.zoo.hclt.learn_hclt_binary

Categorical HCLT
----------------

.. autofunction:: spflow.zoo.hclt.learn_hclt_categorical
