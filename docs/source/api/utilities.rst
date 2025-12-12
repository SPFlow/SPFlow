Utilities
=========

Helper functions and utilities for model visualization, I/O, and analysis.

Visualization
-------------

Visualize probabilistic circuit structures as graphs.

.. autofunction:: spflow.utils.visualization.visualize

Model I/O
---------

Save and load models from disk.

.. autofunction:: spflow.utils.model_manager.save_model

.. autofunction:: spflow.utils.model_manager.load_model


Cache
-----

Utilities for caching intermediate computations to speed up inference.

.. autoclass:: spflow.utils.cache.Cache

ImageSamplingContext
--------------------

2D sampling context for image-based probabilistic circuits. Manages sampling
state with explicit height and width dimensions for convolutional layers.

.. autoclass:: spflow.utils.image_sampling_context.ImageSamplingContext