Learning Algorithms
===================

SPFlow provides multiple algorithms for learning Sum-Product Networks, including structure learning and parameter learning.

Structure Learning
------------------

LearnSPN Algorithm
^^^^^^^^^^^^^^^^^^

The LearnSPN algorithm automatically learns SPN structure from data using recursive partitioning.

.. currentmodule:: spflow.learn.learn_spn

.. autofunction:: learn_spn

.. autoclass:: LearnSPN
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Parameter Learning
------------------

Gradient Descent
^^^^^^^^^^^^^^^^

Train SPN parameters using gradient-based optimization with PyTorch's autograd.

.. currentmodule:: spflow.learn.gradient_descent

.. autofunction:: train_gradient_descent

.. autoclass:: GradientDescentTrainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Expectation-Maximization
^^^^^^^^^^^^^^^^^^^^^^^^^

Train SPN parameters using the EM algorithm with closed-form updates.

.. currentmodule:: spflow.learn.expectation_maximization

.. autofunction:: expectation_maximization

.. autoclass:: EMTrainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Algorithm Comparison
--------------------

+------------------+------------------------+----------------------+-------------------------+
| Algorithm        | Speed                  | Convergence          | Use Case                |
+==================+========================+======================+=========================+
| LearnSPN         | Fast (structure)       | Heuristic-based      | Automatic structure     |
|                  |                        |                      | discovery               |
+------------------+------------------------+----------------------+-------------------------+
| Gradient Descent | Medium-Slow            | Local optima         | Deep models, large      |
|                  | (depends on optimizer) | (gradient-based)     | datasets, GPU training  |
+------------------+------------------------+----------------------+-------------------------+
| EM               | Fast                   | Local optima         | Simple models,          |
|                  |                        | (closed-form)        | interpretability        |
+------------------+------------------------+----------------------+-------------------------+

Usage Examples
--------------

Structure Learning
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow.learn import learn_spn
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope

   scope = Scope(list(range(5)))
   leaf_layer = Normal(scope=scope, out_channels=4)

   model = learn_spn(
       data,
       leaf_modules=leaf_layer,
       out_channels=1,
       min_instances_slice=100
   )

Gradient Descent Training
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from spflow.learn import train_gradient_descent

   trained_model = train_gradient_descent(
       model,
       train_data,
       learning_rate=0.01,
       num_epochs=100,
       batch_size=128
   )

EM Training
^^^^^^^^^^^

.. code-block:: python

   from spflow.learn import expectation_maximization

   trained_model = expectation_maximization(
       model,
       data,
       num_iterations=20
   )
