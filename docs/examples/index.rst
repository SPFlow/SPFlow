Examples Gallery
================

This gallery showcases practical examples of using SPFlow for various tasks. Each example demonstrates specific features and use cases.

.. note::
   These examples are based on the code snippets in the README and demonstrate real-world applications of SPFlow.

Basic Examples
--------------

Manual SPN Construction
^^^^^^^^^^^^^^^^^^^^^^^^

Build a simple Sum-Product Network from scratch:

.. code-block:: python

   import torch
   from spflow.modules import Sum, Product
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow import log_likelihood

   # Create leaf layer for 2 features
   scope = Scope([0, 1])
   leaf_layer = Normal(scope=scope, out_channels=4)

   # Combine with product and sum nodes
   product = Product(inputs=leaf_layer)
   model = Sum(inputs=product, out_channels=2)

   # Compute log-likelihood
   data = torch.randn(32, 2)
   ll = log_likelihood(model, data)
   print(f"Log-likelihood shape: {ll.shape}")

**Use case**: Manual model design, custom architectures

Density Estimation
^^^^^^^^^^^^^^^^^^

Learn a density model from data:

.. code-block:: python

   import torch
   from spflow.learn import learn_spn
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow import log_likelihood

   # Generate synthetic data
   torch.manual_seed(42)
   data = torch.randn(1000, 5)

   # Learn structure
   scope = Scope(list(range(5)))
   leaf_layer = Normal(scope=scope, out_channels=4)
   model = learn_spn(
       data,
       leaf_modules=leaf_layer,
       out_channels=1,
       min_instances_slice=100
   )

   # Evaluate on test data
   test_data = torch.randn(100, 5)
   ll = log_likelihood(model, test_data)
   print(f"Average test log-likelihood: {ll.mean():.4f}")

**Use case**: Unsupervised density estimation, anomaly detection

Advanced Examples
-----------------

High-Dimensional Data with RAT-SPN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scale to high-dimensional data efficiently:

.. code-block:: python

   import torch
   from spflow.modules.rat import RatSPN
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow import log_likelihood

   # High-dimensional data (e.g., image features)
   num_features = 256
   scope = Scope(list(range(num_features)))
   leaf_layer = Normal(scope=scope, out_channels=8, num_repetitions=4)

   # Create RAT-SPN
   model = RatSPN(
       leaf_modules=[leaf_layer],
       n_root_nodes=1,
       n_region_nodes=16,
       num_repetitions=4,
       depth=4,
       outer_product=False
   )

   # Process batch
   data = torch.randn(64, num_features)
   ll = log_likelihood(model, data)
   print(f"Batch log-likelihood: {ll.mean():.4f}")

**Use case**: Image modeling, high-dimensional density estimation

Conditional Generation
^^^^^^^^^^^^^^^^^^^^^^

Sample from conditional distributions:

.. code-block:: python

   import torch
   from spflow.modules import Sum
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope, SamplingContext
   from spflow import sample

   # Build model
   scope = Scope([0, 1, 2, 3])
   leaf_layer = Normal(scope=scope, out_channels=4)
   model = Sum(inputs=leaf_layer, out_channels=1)

   # Conditional sampling: observe features 0,1, sample 2,3
   n_samples = 10
   evidence = torch.zeros(n_samples, 4)
   evidence[:, 0] = 1.0  # Fix feature 0
   evidence[:, 1] = -1.0  # Fix feature 1
   evidence[:, 2:] = torch.nan  # Sample features 2,3

   # Set up sampling context
   channel_index = torch.zeros(n_samples, 4, dtype=torch.int64)
   mask = torch.isnan(evidence)
   ctx = SamplingContext(channel_index=channel_index, mask=mask)

   # Generate conditional samples
   samples = sample(model, evidence, sampling_ctx=ctx)
   print(f"Conditional samples:\n{samples[:, 2:]}")  # Show sampled features

**Use case**: Conditional generation, missing data imputation

Mixed Data Types
^^^^^^^^^^^^^^^^

Handle both continuous and discrete features:

.. code-block:: python

   import torch
   from spflow.modules import Sum, Product
   from spflow.modules.leaf import Normal, Categorical
   from spflow.meta import Scope

   # Continuous features: 0, 1
   continuous_leaf = Normal(scope=Scope([0, 1]), out_channels=4)

   # Discrete feature: 2 (10 categories)
   discrete_leaf = Categorical(scope=Scope([2]), out_channels=4, K=10)

   # Combine with product (factorized)
   product = Product(inputs=[continuous_leaf, discrete_leaf])
   model = Sum(inputs=product, out_channels=1)

   # Mixed data
   data_continuous = torch.randn(100, 2)
   data_discrete = torch.randint(0, 10, (100, 1)).float()
   data = torch.cat([data_continuous, data_discrete], dim=1)

   from spflow import log_likelihood
   ll = log_likelihood(model, data)
   print(f"Log-likelihood for mixed data: {ll.mean():.4f}")

**Use case**: Tabular data with mixed types, real-world datasets

Parameter Learning
------------------

Gradient Descent Training
^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize parameters using gradient-based methods:

.. code-block:: python

   import torch
   from spflow.modules import Sum
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow.learn import train_gradient_descent

   # Create model
   scope = Scope([0, 1, 2])
   leaf_layer = Normal(scope=scope, out_channels=8)
   model = Sum(inputs=leaf_layer, out_channels=1)

   # Training data
   train_data = torch.randn(5000, 3)

   # Train with gradient descent
   trained_model = train_gradient_descent(
       model,
       train_data,
       learning_rate=0.01,
       num_epochs=100,
       batch_size=128
   )

   # Evaluate
   from spflow import log_likelihood
   ll = log_likelihood(trained_model, train_data)
   print(f"Training log-likelihood: {ll.mean():.4f}")

**Use case**: Large-scale training, deep probabilistic models

Expectation-Maximization
^^^^^^^^^^^^^^^^^^^^^^^^^

Use EM for efficient parameter learning:

.. code-block:: python

   import torch
   from spflow.modules import Sum
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow.learn import expectation_maximization

   # Create model
   scope = Scope([0, 1])
   leaf_layer = Normal(scope=scope, out_channels=4)
   model = Sum(inputs=leaf_layer, out_channels=1)

   # Training data
   data = torch.randn(1000, 2)

   # Train with EM
   trained_model = expectation_maximization(
       model,
       data,
       num_iterations=20
   )

   from spflow import log_likelihood
   ll = log_likelihood(trained_model, data)
   print(f"Final log-likelihood: {ll.mean():.4f}")

**Use case**: Traditional EM-based learning, interpretable models

Visualization
-------------

Model Structure Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the learned structure:

.. code-block:: python

   import torch
   from spflow.modules import Sum, Product
   from spflow.modules.leaf import Normal, Categorical
   from spflow.meta import Scope
   from spflow.utils.visualization import visualize_module

   # Build a small model
   leaf1 = Normal(scope=Scope([0, 1]), out_channels=2)
   leaf2 = Categorical(scope=Scope([2]), out_channels=2, K=5)
   product = Product(inputs=[leaf1, leaf2])
   model = Sum(inputs=product, out_channels=1)

   # Visualize
   visualize_module(
       model,
       output_path="my_spn_structure",
       show_scope=True,
       show_shape=True,
       show_params=True,
       format="png"
   )
   print("Visualization saved to my_spn_structure.png")

**Use case**: Understanding model structure, debugging, presentations

Utilities
---------

Model Saving and Loading
^^^^^^^^^^^^^^^^^^^^^^^^^

Persist trained models:

.. code-block:: python

   import torch
   from spflow.modules import Sum
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow.utils.model_manager import save_model, load_model

   # Create and train model
   scope = Scope([0, 1])
   model = Sum(inputs=Normal(scope=scope, out_channels=4), out_channels=1)

   # Train model here...

   # Save
   save_model(model, "trained_spn.pt")
   print("Model saved")

   # Load
   loaded_model = load_model("trained_spn.pt")
   print("Model loaded")

   # Use loaded model
   from spflow import log_likelihood
   data = torch.randn(10, 2)
   ll = log_likelihood(loaded_model, data)

**Use case**: Model persistence, deployment

Next Steps
----------

- Review the :doc:`../tutorials/index` for detailed walkthroughs
- Explore the :doc:`../api/index` for complete API reference
- Check the :doc:`../architecture` guide for design patterns
