===================
Continuous Mixtures
===================

SPFlow supports **continuous mixtures of tractable probabilistic models**, inspired by:

*Correia et al., "Continuous Mixtures of Tractable Probabilistic Models" (2023).*

The core idea is to introduce a low-dimensional latent variable ``z`` and a decoder network
``φ(z)`` that outputs parameters of a tractable model. The marginal density is an integral

.. math::

    p(x) = \mathbb{E}_{p(z)}[p(x \mid \phi(z))] = \int p(x \mid \phi(z)) p(z)\,dz

SPFlow approximates the integral with **Sobol-RQMC** integration points, and then compiles the
result into a standard SPFlow :class:`spflow.modules.module.Module` (a discrete mixture / ``Sum``)
so that you can use the normal inference API (``log_likelihood``, ``sample``, marginalization).

Two structures are provided, matching the paper setup:

- ``S_F``: fully factorized (independent per feature)
- ``S_CLT``: Chow–Liu tree structure (discrete only in v1)


Factorized Continuous Mixtures (``S_F``)
========================================

Use :func:`spflow.learn.learn_continuous_mixture_factorized` for factorized continuous mixtures.
Supported leaves:

- Bernoulli (binary data, values in ``{0, 1}``, NaNs allowed)
- Categorical(K) (integer-coded data in ``{0, ..., K-1}``, NaNs allowed)
- Normal (continuous data, NaNs allowed)

Example (Bernoulli)
-------------------

.. code-block:: python

    import torch
    from spflow.learn import learn_continuous_mixture_factorized
    from spflow.learn.continuous_mixtures import LatentOptimizationConfig

    # (N, F) with values in {0, 1}; NaNs are treated as missing.
    data = torch.randint(0, 2, (2048, 50), dtype=torch.float32)
    data[:100, 0] = float("nan")

    model = learn_continuous_mixture_factorized(
        data,
        leaf="bernoulli",
        latent_dim=4,
        num_points_train=256,
        num_points_eval=256,
        lo=LatentOptimizationConfig(enabled=True, num_points=64),
    )

    ll = model.log_likelihood(data)  # (N, 1, 1, 1)


Chow–Liu Continuous Mixtures (``S_CLT``)
========================================

Use :func:`spflow.learn.learn_continuous_mixture_cltree` to learn continuous mixtures whose
components are Chow–Liu trees.

Important notes:

- **Discrete only**: Bernoulli / Categorical(K)
- **Complete data required** (no NaNs) for structure learning and training
- Output is wrapped so that ``log_likelihood`` returns a single feature (joint score).
  This uses :class:`spflow.modules.wrapper.joint.JointLogLikelihood`, which simply **sums**
  the per-feature log-likelihood contributions into a joint score (it is a tensor reduction,
  not a factorization/product semantics change).

Example (Categorical)
---------------------

.. code-block:: python

    import torch
    from spflow.learn import learn_continuous_mixture_cltree
    from spflow.learn.continuous_mixtures import LatentOptimizationConfig

    # (N, F) with values in {0, ..., K-1}; must be complete (no NaNs).
    K = 4
    data = torch.randint(0, K, (2048, 50), dtype=torch.float32)

    model = learn_continuous_mixture_cltree(
        data,
        leaf="categorical",
        num_cats=K,
        latent_dim=4,
        num_points_train=256,
        num_points_eval=256,
        lo=LatentOptimizationConfig(enabled=True, num_points=64),
    )

    ll = model.log_likelihood(data)  # (N, 1, 1, 1)
