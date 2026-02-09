================================
Sum of Compatible Squares (SOCS)
================================

SPFlow includes an implementation of **SOCS / Σ2cmp** ("Sum of Compatible Squares Circuits").
SOCS turns a set of (possibly signed) *compatible* component circuits into a valid, non-negative
probability model.

Reference
---------

The core SOCS / Σ2cmp construction is described in the AAAI paper:

- `Sum of Squares Circuits (AAAI) <https://ojs.aaai.org/index.php/AAAI/article/view/34100>`_

Definition
----------

Let ``c_i(x)`` be real-valued component circuits that share the same structured decomposition
("compatible" components).
SOCS defines the non-negative function:

.. math::

   c(x) = \sum_{i=1}^r c_i(x)^2

and the normalized density:

.. math::

   p(x) = \frac{c(x)}{Z}, \qquad Z = \int c(x) \, dx = \sum_{i=1}^r \int c_i(x)^2 \, dx.

In SPFlow, SOCS is implemented as the wrapper module :class:`spflow.zoo.sos.SOCS`.

Why "signed" components?
------------------------

Standard SPFlow sum nodes (:class:`spflow.modules.sums.Sum`) represent convex mixtures and require
strictly positive weights.
To represent signed circuits, the Paper Zoo provides :class:`spflow.zoo.sos.SignedSum`, which allows
**real-valued (including negative) weights**.

Important: ``SignedSum`` is *not* a probabilistic mixture node (its output may be negative), so it
does not implement ``log_likelihood``. Instead, SOCS evaluates signed components using a stable
signed representation internally.

Exact normalization via inner products
--------------------------------------

SOCS needs the normalization terms:

.. math::

   Z_i = \int c_i(x)^2 \, dx.

Rather than building an explicit "squared circuit", SPFlow computes these terms using an exact,
bottom-up **inner-product dynamic program** implemented in :mod:`spflow.utils.inner_product`.

The implementation supports exact inner products for common leaves (and can be extended by adding
new closed-form formulas in ``spflow/exp/sos/inner_product.py``). Currently supported include:

- ``Normal``, ``Bernoulli``, ``Categorical``
- ``Exponential``, ``Laplace``, ``LogNormal``
- ``Poisson``, ``Gamma``
- ``CLTree`` (only when both trees share the same structure)

Non-scalar outputs
------------------

SPFlow modules return log-likelihood tensors with shape ``(batch, F, C, R)`` where:

- ``F`` = output features
- ``C`` = output channels
- ``R`` = repetitions

SOCS supports non-scalar component outputs and normalizes **per output entry**:

.. math::

   Z[f,c,r] = \sum_i \int c_{i,f,c,r}(x)^2 \, dx.

This makes SOCS usable as a building block inside larger architectures (e.g., class-conditional
SOCS with ``channels = num_classes``).

Sampling (signed components)
----------------------------

Sampling from the density proportional to ``c_i(x)^2`` is not generally tractable with the standard
top-down sampler when ``c_i`` is signed.

SPFlow implements an **independence Metropolis–Hastings** sampler for scalar-output SOCS
(``out_shape == (1,1,1)``), targeting:

.. math::

   \pi_i(x) \propto c_i(x)^2.

For signed components, the proposal distribution is built by replacing each ``SignedSum`` node with
a standard ``Sum`` using **abs(weights)** (normalized). This yields a monotone proposal circuit
``q_i(x)`` that supports both ``sample()`` and ``log_likelihood()``.

The MH kernel uses the acceptance rule:

.. math::

   \alpha(x \to x') = \min\left(1, \frac{\pi_i(x') \, q_i(x)}{\pi_i(x) \, q_i(x')}\right).

Configuration
-------------

Sampling behavior can be controlled via :class:`spflow.utils.cache.Cache` extras:

- ``cache.extras["socs_mcmc_steps"]``: number of MH steps after burn-in (default: 50)
- ``cache.extras["socs_mcmc_burn_in"]``: burn-in steps (default: 10)

Limitations
-----------

- SOCS sampling currently supports **unconditional** sampling only (no evidence / NaN-conditional sampling).
- SOCS sampling currently supports only **scalar outputs** (``out_shape == (1,1,1)``).

Compatibility checks
--------------------

SOCS assumes component circuits are compatible (same decomposition / region graph).
SPFlow provides conservative structural checks in :mod:`spflow.utils.compatibility`:

- :func:`spflow.zoo.sos.check_compatible_components`
- :func:`spflow.zoo.sos.check_socs_compatibility`

These utilities verify that corresponding nodes across components have the same "skeleton"
(node types, scopes, arities, and selected structural metadata like ``Cat.dim`` and ``CLTree.parents``).

Structure builder
-----------------

To reduce boilerplate, SPFlow includes a small builder that clones a template circuit into multiple
compatible components and optionally converts all ``Sum`` nodes to ``SignedSum`` nodes:

- :func:`spflow.zoo.sos.build_socs`

Minimal example
---------------

Build a SOCS model from one signed component and evaluate it::

    import torch
    from spflow.meta.data.scope import Scope
    from spflow.modules.leaves import Bernoulli
    from spflow.zoo.sos import SignedSum, SOCS

    b1 = Bernoulli(scope=Scope([0]), probs=torch.tensor([[[0.2]]]))
    b2 = Bernoulli(scope=Scope([0]), probs=torch.tensor([[[0.8]]]))
    comp = SignedSum(inputs=[b1, b2], weights=torch.tensor([[[[0.9]], [[-0.2]]]]))

    model = SOCS([comp])
    x = torch.tensor([[0.0], [1.0]])
    ll = model.log_likelihood(x)  # (B,1,1,1)


API Reference
-------------


.. autoclass:: spflow.zoo.sos.SOCS
   :members:
   :show-inheritance:

.. autoclass:: spflow.zoo.sos.SignedSum
   :members:
   :show-inheritance:

Builders
--------

.. autofunction:: spflow.zoo.sos.build_socs

.. autofunction:: spflow.zoo.sos.build_abs_weight_proposal

Compatibility
-------------

.. autofunction:: spflow.zoo.sos.check_compatible_components

.. autofunction:: spflow.zoo.sos.check_socs_compatibility

Exact Inner Products
--------------------

.. autofunction:: spflow.zoo.sos.inner_product_matrix

.. autofunction:: spflow.zoo.sos.leaf_inner_product

.. autofunction:: spflow.zoo.sos.log_self_inner_product_scalar

Signed Semiring Utilities
-------------------------

.. autofunction:: spflow.zoo.sos.signed_logsumexp

.. autofunction:: spflow.zoo.sos.sign_of

.. autofunction:: spflow.zoo.sos.logabs_of
