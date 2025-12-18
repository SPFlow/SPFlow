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

MarginalizationContext
----------------------

Context for spatial marginalization in image data.

.. autoclass:: spflow.modules.wrapper.image_wrapper.MarginalizationContext
   :members:
   :undoc-members:
