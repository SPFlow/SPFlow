Exceptions
==========

Custom exception classes used throughout SPFlow.

.. currentmodule:: spflow.exceptions

Exception Classes
-----------------

InvalidParameterCombinationError
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Raised when invalid parameter combinations are provided.

.. autoexception:: InvalidParameterCombinationError
   :members:
   :show-inheritance:

ScopeError
^^^^^^^^^^

Raised when there are scope-related errors (e.g., invalid scope operations, incompatible scopes).

.. autoexception:: ScopeError
   :members:
   :show-inheritance:

GraphvizError
^^^^^^^^^^^^^

Raised when GraphViz visualization fails.

.. autoexception:: GraphvizError
   :members:
   :show-inheritance:

Usage Examples
--------------

Handling ScopeError
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow import ScopeError
   from spflow.meta import Scope
   from spflow.modules import Product

   try:
       # Attempting to create Product with overlapping scopes
       # (Products expect disjoint scopes)
       leaf1 = Normal(scope=Scope([0, 1]), out_channels=4)
       leaf2 = Normal(scope=Scope([1, 2]), out_channels=4)
       product = Product(inputs=[leaf1, leaf2])
   except ScopeError as e:
       print(f"Scope error: {e}")

Handling InvalidParameterCombinationError
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow import InvalidParameterCombinationError
   from spflow.modules.leaf import Normal

   try:
       # Invalid parameter combination
       leaf = Normal(
           scope=[0],
           out_channels=4,
           mean="invalid_type"  # Should be float or tensor
       )
   except InvalidParameterCombinationError as e:
       print(f"Invalid parameters: {e}")

Handling GraphvizError
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow import GraphvizError
   from spflow.utils.visualization import visualize_module

   try:
       visualize_module(
           model,
           output_path="/invalid/path/structure",
           format="png"
       )
   except GraphvizError as e:
       print(f"Visualization failed: {e}")
       print("Make sure GraphViz is installed: pip install graphviz")
