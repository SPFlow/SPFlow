Modules (Inner Nodes)
=====================

This page documents the core module classes in SPFlow, including the base module class and inner node types (Sum and Product).

Base Module
-----------

.. currentmodule:: spflow.modules.module

.. autoclass:: Module
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em

Sum Nodes
---------

Sum nodes represent weighted mixtures of child distributions.

.. currentmodule:: spflow.modules.sum

.. autoclass:: Sum
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.marginalize

Elementwise Sum
^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.elementwise_sum

.. autoclass:: ElementwiseSum
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.marginalize

Product Nodes
-------------

Product nodes represent factorizations (independence assumptions).

Base Product
^^^^^^^^^^^^

.. currentmodule:: spflow.modules.base_product

.. autoclass:: BaseProduct
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.marginalize

Product
^^^^^^^

.. currentmodule:: spflow.modules.product

.. autoclass:: Product
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.marginalize

Elementwise Product
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.elementwise_product

.. autoclass:: ElementwiseProduct
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood

Outer Product
^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.outer_product

.. autoclass:: OuterProduct
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood

Factorization
^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.factorize

.. autoclass:: Factorize
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.marginalize

RAT-SPN Architecture
--------------------

RAT-SPN (Randomized and Tensorized Sum-Product Network) is an efficient architecture for high-dimensional data.

.. currentmodule:: spflow.modules.rat.rat_spn

.. autoclass:: RatSPN
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample

RAT Mixing Layer
^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.rat.rat_mixing_layer

.. autoclass:: RatMixingLayer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample

Operations
----------

Split operations for partitioning data.

Split
^^^^^

.. currentmodule:: spflow.modules.ops.split

.. autoclass:: Split
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.sample
   spflow.marginalize

Split Halves
^^^^^^^^^^^^

.. currentmodule:: spflow.modules.ops.split_halves

.. autoclass:: SplitHalves
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.marginalize

Split Alternate
^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.ops.split_alternate

.. autoclass:: SplitAlternate
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.marginalize

Cat
^^^

.. currentmodule:: spflow.modules.ops.cat

.. autoclass:: Cat
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.marginalize

Wrappers
--------

Wrapper modules for specialized data types.

Abstract Wrapper
^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.wrapper.abstract_wrapper

.. autoclass:: AbstractWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Image Wrapper
^^^^^^^^^^^^^

.. currentmodule:: spflow.modules.wrapper.ImageWrapper

.. autoclass:: ImageWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. rubric:: Dispatched Methods

.. autosummary::
   :nosignatures:

   spflow.log_likelihood
   spflow.sample
   spflow.em
   spflow.marginalize
