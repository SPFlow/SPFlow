Continuous Mixtures (CMs)
=========================

Continuous Mixtures of Tractable Probabilistic Models introduce a low-dimensional latent variable and a decoder network that outputs parameters of a tractable model, allowing for continuous variations in the model structure.

Reference
---------

Continuous Mixtures are described in:

- `Continuous Mixtures of Tractable Probabilistic Models (AAAI 2023) <https://arxiv.org/abs/2209.11718>`_

Overview
--------

The marginal density of a continuous mixture is an integral over the latent space:

.. math::

    p(x) = \mathbb{E}_{p(z)}[p(x \mid \phi(z))] = \int p(x \mid \phi(z)) p(z)\,dz

SPFlow approximates this integral using **Sobol-RQMC** (Randomized Quasi-Monte Carlo) points and then compiles the result into a standard SPFlow module for inference.

Key features:
~~~~~~~~~~~~~

- **Latent Optimization (LO)**: Supports optimizing latent variables for better data fit.
- **Discrete compilation**: Compiled circuits can be used with all standard SPFlow operations.
- **Multiple structures**: Supports both factorized (independent) and Chow-Liu tree structures for the components.

Implementation
--------------

Factorized Continuous Mixtures
------------------------------

.. autofunction:: spflow.zoo.cms.learn_continuous_mixture_factorized

Chow–Liu Continuous Mixtures
----------------------------

.. autofunction:: spflow.zoo.cms.learn_continuous_mixture_cltree
