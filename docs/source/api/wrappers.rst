Wrapper Modules
===============

Wrapper classes that adapt SPFlow modules for specific data formats and use cases.

The wrapper pattern enables flexible extension of SPFlow capabilities while
maintaining compatibility with the core module interfaces.

Wrapper (Base Class)
--------------------

Abstract base class for SPFlow module wrappers.

.. autoclass:: spflow.modules.wrapper.base.Wrapper
   :members:
   :undoc-members:
   :show-inheritance:

ImageWrapper
------------

Adapts SPFlow modules for 4D image data (batch, channels, height, width).
Provides automatic conversion between flattened tensors and image format.

.. autoclass:: spflow.modules.wrapper.image_wrapper.ImageWrapper
   :members:
   :undoc-members:
   :show-inheritance:

JointLogLikelihood
------------------

Wrapper that exposes the **joint** log-likelihood as a single feature, i.e. reduces a
``(batch, features, channels, repetitions)`` tensor to ``(batch, 1, channels, repetitions)``
by summing over the feature axis.

This is a convenience adapter to make some multivariate leaves (e.g. ``CLTree``) behave like
typical "root" modules in SPFlow. It is a tensor reduction and **does not** introduce any
additional independence/factorization semantics.

.. autoclass:: spflow.zoo.cms.JointLogLikelihood
   :members:
   :undoc-members:
   :show-inheritance:

MarginalizationContext
----------------------

Context for spatial marginalization in image data.

.. autoclass:: spflow.modules.wrapper.image_wrapper.MarginalizationContext
   :members:
   :undoc-members:
