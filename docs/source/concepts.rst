========
Concepts
========

This page is a stable, linkable reference for SPFlow concepts (separate from the API reference and notebooks).

See also :doc:`SOCS <exp/socs>` for details on signed circuits and compatibility.

.. _concepts-shapes-and-dimensions:

Shapes and Dimensions
=====================

SPFlow modules use a consistent internal shape convention: **(features, channels, repetitions)**.
You will often see this displayed as ``D`` (features), ``C`` (channels), and ``R`` (repetitions) in
``model.to_str()`` output.

Terminology
-----------

- **Features (D)**: Number of random variables represented by the module (usually ``len(scope)``).
- **Channels (C)**: Parallel distributions/mixture channels computed in one forward pass.
- **Repetitions (R)**: Independent parameterizations of the same structure.

Where shapes appear
-------------------

- **Input data**: ``data`` is shaped ``(batch, num_features)``.
- **Log-likelihood outputs**: ``log_likelihood`` returns ``(batch, out_features, out_channels)``.
- **Module metadata**: :class:`spflow.modules.module_shape.ModuleShape` stores ``(features, channels, repetitions)``.

Practical tips
--------------

- Use ``model.to_str()`` to sanity-check shapes and scopes end-to-end.
- If ``data.shape[1] != len(model.scope)``, check your leaf scopes first.

Related references
------------------

- :ref:`concepts-scopes-and-decomposability`
- :doc:`API: ModuleShape <api/module_shape>`
- :doc:`API: Scope <api/scope>`

.. _concepts-scopes-and-decomposability:

Scopes and Decomposability
==========================

A **scope** identifies which input variables (features) a module operates on.
Scopes enforce the structural constraints that make inference tractable.

What is a scope?
----------------

Use :class:`spflow.meta.Scope` to describe feature indices::

    from spflow.meta import Scope

    scope = Scope([0, 1, 2])

Rules of thumb
--------------

- **Sum nodes** combine inputs with the **same** scope.
- **Product nodes** combine inputs with **disjoint** scopes (decomposability / independence assumption).

Common failure modes
--------------------

- **Scope mismatch in a Sum**: you mixed modules that do not cover the same variables.
- **Overlapping scopes in a Product**: you combined two modules that both model the same feature(s).

Related references
------------------

- :doc:`API: Scope <api/scope>`
- :ref:`concepts-shapes-and-dimensions`

.. _concepts-missing-data-and-evidence:

Missing Data and Evidence
=========================

SPFlow uses **NaN-based evidence**: missing values are represented with ``torch.nan``.
This makes it easy to mix observed and unobserved variables in the same tensor.

Log-likelihood with missing data
--------------------------------

When computing likelihoods, NaN entries are treated as "unknown" variables to marginalize out::

    import torch

    data = torch.randn(32, 5)
    data[0, 2] = float("nan")  # feature 2 missing for sample 0

    log_ll = model.log_likelihood(data)

Conditional sampling with evidence
----------------------------------

For conditional sampling, you can provide an evidence tensor where NaNs indicate values to sample::

    evidence = torch.full((10, num_features), float("nan"))
    evidence[:, 0] = 0.5  # condition on feature 0

    samples = model.sample_with_evidence(evidence=evidence)

Related references
------------------

- :doc:`FAQ <faq>`
- :doc:`API Reference <api/index>`

.. _concepts-differentiable-sampling:

Differentiable Sampling
=======================

Differentiable sampling support has been removed from SPFlow.

Use ``sample`` for stochastic generation and ``mpe`` for deterministic decoding.
For APC models, latent-stat extraction and KL-style training helpers are currently unavailable.

.. _concepts-caching-and-dispatch:

Caching and Dispatch
====================

Probabilistic circuits are DAGs, and many operations reuse subcomputations.
SPFlow provides a lightweight caching mechanism to avoid redundant work during inference, learning, and sampling.

Cache basics
------------

- Use :class:`spflow.utils.cache.Cache` to memoize intermediate results across a single traversal.
- Many modules use the :func:`spflow.utils.cache.cached` decorator for operations like ``log_likelihood``.
- :class:`spflow.utils.cache.Cache` also provides ``Cache.extras`` for storing custom, user-defined information that
  should be available throughout a recursive traversal.

When you should care
--------------------

- Repeatedly calling ``log_likelihood`` on the same model inside a loop can be faster if you reuse a cache.
- Debugging unexpected values is easier if you can control whether cached results are reused.

Related references
------------------

- :class:`spflow.utils.cache.Cache`
- :func:`spflow.utils.cache.cached`
