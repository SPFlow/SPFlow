=====================================
Probabilistic Integral Circuits (PICs)
=====================================

Probabilistic Integral Circuits (PICs) are a framework for scaling continuous latent variable models by representing them as integrals over tractable circuits. They allow for flexible neural functional sharing while maintaining tractability via quadrature-based materialization.

Reference
---------

PICs are described in the NeurIPS 2024 paper:

- `Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits <https://arxiv.org/abs/2410.02410>`_

Overview
--------

A PIC is a symbolic representation of a continuous mixture model. It consists of:

- **Input Units**: Representing leaf functions :math:`f_u(X_u, Z_u)`.
- **Sum Units**: Representing convex mixtures.
- **Product Units**: Representing factorized distributions.
- **Integral Units**: Representing integration over continuous latent variables :math:`Z`.

Pipeline
--------

The typical PIC workflow in SPFlow follows these steps:

1. **RegionGraph to PIC**: Convert a standard :class:`spflow.meta.region_graph.RegionGraph` into a symbolic PIC using :func:`spflow.zoo.pic.rg2pic`.
2. **Functional Sharing**: Attach neural networks (e.g., :class:`spflow.zoo.pic.SharedMLP`) to Integral units to allow parameters to be shared across the circuit.
3. **Materialization (PIC to QPC)**: Convert the symbolic PIC into a materialized **Quadrature Product Circuit (QPC)** using :func:`spflow.zoo.pic.pic2qpc`. The QPC is a standard SPFlow :class:`spflow.modules.Module` that can be evaluated using ``log_likelihood()``.

Merge Strategies
----------------

When converting a RegionGraph to a PIC, SPFlow supports different merge strategies:

- ``AUTO``: Uses Tucker merge if latent variables differ, otherwise CP merge (matching paper semantics).
- ``TUCKER``: Always uses Tucker-style merging (resulting in :class:`spflow.modules.products.OuterProduct` after materialization).
- ``CP``: Always uses CP-style merging (resulting in :class:`spflow.modules.products.ElementwiseProduct` after materialization).

Functional Sharing
------------------

PICs allow parameters of the circuit to be computed by neural networks. This is implemented via:

- :class:`spflow.zoo.pic.SharedMLP`: A simple MLP shared across all units in a group.
- :class:`spflow.zoo.pic.MultiHeadedMLP`: An MLP with multiple output heads for different units.
- :class:`spflow.zoo.pic.FourierFeatures`: Positional encoding for latent variables.

Tensorized QPCs
---------------

For high-performance evaluation, PICs can be materialized into a **Tensorized QPC**. This mode avoids creating thousands of small SPFlow modules and instead uses a single folded module that performs evaluation using efficient tensor operations.

- :class:`spflow.zoo.pic.TensorizedQPC`: A folded module for efficient PIC inference.

Minimal Example
---------------

.. code-block:: python

    import torch
    from spflow.meta.region_graph import RegionGraph
    from spflow.zoo.pic import rg2pic, pic2qpc, QuadratureRule, PICInput

    # 1. Define a simple RegionGraph
    rg = RegionGraph.from_nested_list([[0, 1]])

    # 2. Define a leaf factory
    class MyInput(PICInput):
        def __init__(self, scope, latent_scope):
            self.scope = scope
            self.latent_scope = latent_scope
        def materialize(self, quadrature_rule):
            # Return a standard SPFlow module (e.g. Gaussian leaves)
            from spflow.modules.leaves import Normal
            K = quadrature_rule.points.shape[0]
            return Normal(scope=self.scope, channels=K)

    # 3. Build symbolic PIC
    pic = rg2pic(rg, leaf_factory=lambda x, z: MyInput(x, z))

    # 4. Materialize to QPC
    q_rule = QuadratureRule(
        points=torch.linspace(-3, 3, 5),
        weights=torch.ones(5) / 5
    )
    model = pic2qpc(pic, q_rule)

    # 5. Evaluate
    x = torch.randn(10, 2)
    ll = model.log_likelihood(x)

API Reference
-------------

Pipeline
--------

.. autofunction:: spflow.zoo.pic.rg2pic

.. autofunction:: spflow.zoo.pic.pic2qpc

.. autoclass:: spflow.zoo.pic.MergeStrategy
   :members:

.. autoclass:: spflow.zoo.pic.QuadratureRule
   :members:

Integral and Sum Units
----------------------

.. autoclass:: spflow.zoo.pic.Integral
   :members:
   :show-inheritance:

.. autoclass:: spflow.zoo.pic.PICSum
   :members:
   :show-inheritance:

.. autoclass:: spflow.zoo.pic.PICProduct
   :members:
   :show-inheritance:

.. autoclass:: spflow.zoo.pic.WeightedSum
   :members:
   :show-inheritance:
   :no-index:

Functional Sharing
------------------

.. autoclass:: spflow.zoo.pic.SharedMLP
   :members:
   :show-inheritance:

.. autoclass:: spflow.zoo.pic.MultiHeadedMLP
   :members:
   :show-inheritance:

.. autoclass:: spflow.zoo.pic.FourierFeatures
   :members:
   :show-inheritance:

.. autoclass:: spflow.zoo.pic.FunctionGroup
   :members:
   :show-inheritance:

Tensorized QPC
--------------

.. autoclass:: spflow.zoo.pic.TensorizedQPC
   :members:
   :show-inheritance:

.. autoclass:: spflow.zoo.pic.TensorizedQPCConfig
   :members:
