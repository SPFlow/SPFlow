Utilities
=========

Utility functions and classes for working with SPFlow models.

Visualization
-------------

Module Visualization
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.visualization

.. autofunction:: visualize_module

.. autoclass:: ModuleVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Module Display
^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.module_display

.. autofunction:: module_to_str

.. autoclass:: ModuleDisplayFormatter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Model Management
----------------

Save and Load Models
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.model_manager

.. autofunction:: save_model

.. autofunction:: load_model

.. autoclass:: ModelManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Leaf Utilities
--------------

.. currentmodule:: spflow.utils.leaf

.. automodule:: spflow.utils.leaf
   :members:
   :undoc-members:
   :show-inheritance:

Graph Algorithms
----------------

Connected Components
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.connected_components

.. autofunction:: find_connected_components

.. autoclass:: ConnectedComponentsFinder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Clustering
----------

K-means
^^^^^^^

.. currentmodule:: spflow.utils.kmeans

.. autofunction:: kmeans

.. autoclass:: KMeans
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

K-means (New Implementation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.kmeans_new

.. autofunction:: kmeans_new

.. autoclass:: KMeansNew
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Statistical Utilities
---------------------

Randomized Dependence Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.rdc

.. autofunction:: rdc

Randomized Dependence Coefficient (PyTorch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.randomized_dependency_coefficients_torch

.. autofunction:: rdc_torch

Empirical CDF (PyTorch)
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.empirical_cdf_torch

.. autofunction:: empirical_cdf

Mathematical Utilities
----------------------

Nearest Symmetric Positive Definite Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.nearest_sym_pd

.. autofunction:: nearest_symmetric_positive_definite

Projections
^^^^^^^^^^^

Parameter projection utilities for constrained optimization.

.. currentmodule:: spflow.utils.projections

.. automodule:: spflow.utils.projections
   :members:
   :undoc-members:
   :show-inheritance:

Complex Numbers
^^^^^^^^^^^^^^^

.. currentmodule:: spflow.utils.complex

.. automodule:: spflow.utils.complex
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Visualizing Models
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow.utils.visualization import visualize_module

   # Visualize model structure
   visualize_module(
       model,
       output_path="spn_structure",
       show_scope=True,
       show_shape=True,
       show_params=True,
       format="png"  # or "svg", "pdf"
   )

Saving and Loading Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow.utils.model_manager import save_model, load_model

   # Save trained model
   save_model(model, "trained_spn.pt")

   # Load model
   loaded_model = load_model("trained_spn.pt")

Computing RDC
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from spflow.utils.randomized_dependency_coefficients_torch import rdc_torch

   # Compute randomized dependence coefficient
   data_x = torch.randn(1000, 5)
   data_y = torch.randn(1000, 5)

   rdc_value = rdc_torch(data_x, data_y)
   print(f"RDC: {rdc_value:.4f}")

K-means Clustering
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from spflow.utils.kmeans import kmeans

   data = torch.randn(1000, 10)
   centers, labels = kmeans(data, k=5, max_iterations=100)

   print(f"Cluster centers shape: {centers.shape}")
   print(f"Labels shape: {labels.shape}")
