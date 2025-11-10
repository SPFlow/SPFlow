Quick Start Guide
=================

This guide will help you get started with SPFlow by walking through several basic examples.

Core Concepts
-------------

Before diving into examples, let's understand the key concepts in SPFlow:

- **Modules:** SPFlow models are built from modules (Sum, Product, Leaf nodes) that inherit from ``torch.nn.Module``
- **Scope:** Each module has a scope defining which features (variables) it operates on
- **Dispatched Functions:** SPFlow uses a dispatch system for polymorphic functions like ``log_likelihood``, ``sample``, and ``em``
- **Channels:** Multiple copies of distributions/nodes that can be combined hierarchically

Example 1: Manual SPN Construction
-----------------------------------

Let's build a simple Sum-Product Network manually:

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

   # Compute log-likelihood on some data
   data = torch.randn(32, 2)  # 32 samples, 2 features
   log_likelihood_output = log_likelihood(model, data)

   print(f"Model:\n{model.to_str()}")
   print(f"Data shape: {data.shape}")
   print(f"Log-likelihood output shape: {log_likelihood_output.shape}")
   print(f"Log-likelihood sample: {log_likelihood_output[0]}")

**Output:**

.. code-block:: text

   Model:
   Sum [D=1, C=2] [weights: (1, 4, 2)] → scope: 0-1
   └─ Product [D=1, C=4] → scope: 0-1
      └─ Normal [D=2, C=4] → scope: 0-1
   Data shape: torch.Size([32, 2])
   Log-likelihood output shape: torch.Size([32, 1, 2])
   Log-likelihood sample: tensor([[-3.1183, -3.8367]], grad_fn=<SelectBackward0>)

Example 2: RAT-SPN (Randomized & Tensorized)
---------------------------------------------

RAT-SPNs are efficient architectures that scale to high-dimensional data:

.. code-block:: python

   import torch
   from spflow.modules.rat import RatSPN
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow import log_likelihood

   # Create leaf layer
   num_features = 64
   scope = Scope(list(range(num_features)))
   leaf_layer = Normal(scope=scope, out_channels=4, num_repetitions=2)

   # Create and use RAT-SPN model
   data = torch.randn(100, num_features)
   model = RatSPN(
       leaf_modules=[leaf_layer],
       n_root_nodes=1,
       n_region_nodes=8,
       num_repetitions=2,
       depth=3,
       outer_product=False
   )

   log_likelihood_output = log_likelihood(model, data)

   print(f"Data shape: {data.shape}")
   print(f"Log-likelihood output shape: {log_likelihood_output.shape}")
   print(f"Log-likelihood - Mean: {log_likelihood_output.mean():.4f}")

**Output:**

.. code-block:: text

   Data shape: torch.Size([100, 64])
   Log-likelihood output shape: torch.Size([100, 1, 1])
   Log-likelihood - Mean: -477.4794

Example 3: Structure Learning with LearnSPN
--------------------------------------------

SPFlow can automatically learn SPN structures from data:

.. code-block:: python

   import torch
   from spflow.learn import learn_spn
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow import log_likelihood

   torch.manual_seed(42)

   # Create leaf layer with Gaussian distributions
   scope = Scope(list(range(5)))
   leaf_layer = Normal(scope=scope, out_channels=4)

   # Learn SPN structure from data
   # Construct synthetic data with three different clusters
   torch.manual_seed(0)
   cluster_1 = torch.randn(200, 5) + torch.tensor([0, 0, 0, 0, 0])
   cluster_2 = torch.randn(200, 5) + torch.tensor([5, 5, 5, 5, 5])
   cluster_3 = torch.randn(200, 5) + torch.tensor([-5, -5, -5, -5, -5])
   data = torch.vstack([cluster_1, cluster_2, cluster_3]).float()

   model = learn_spn(
       data,
       leaf_modules=leaf_layer,
       out_channels=1,
       min_instances_slice=100
   )

   # Use the learned model
   log_likelihood_output = log_likelihood(model, data)

   print(f"Learned model structure:\n{model.to_str()}")
   print(f"Log-likelihood - Mean: {log_likelihood_output.mean():.4f}")

Example 4: Sampling from an SPN
--------------------------------

Generate samples from a trained model:

.. code-block:: python

   import torch
   from spflow.modules import Sum
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope, SamplingContext
   from spflow import log_likelihood, sample

   torch.manual_seed(42)

   # Create a simple SPN with 3 features
   scope = Scope([0, 1, 2])
   leaf_layer = Normal(scope=scope, out_channels=2)
   model = Sum(inputs=leaf_layer, out_channels=1)

   # Sample from the model
   n_samples = 2
   out_features = model.out_features
   evidence = torch.full((n_samples, out_features), torch.nan)  # No conditioning
   channel_index = torch.full((n_samples, out_features), 0, dtype=torch.int64)
   mask = torch.full((n_samples, out_features), True, dtype=torch.bool)
   sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

   samples = sample(model, evidence, sampling_ctx=sampling_ctx)

   # Compute log-likelihood for the samples
   log_likelihood_output = log_likelihood(model, samples)

   print(f"Generated samples:\n{samples}")
   print(f"Log-likelihood of samples: {log_likelihood_output[0]}")

**Output:**

.. code-block:: text

   Generated samples:
   tensor([[ 0.1607,  0.6218, -1.1670],
           [ 0.5427,  0.3331, -0.6964]])
   Log-likelihood of samples: tensor([[-0.4677],
           [-0.5822],
           [-1.4382]], grad_fn=<SelectBackward0>)

Example 5: Training with Gradient Descent
------------------------------------------

Train an SPN using gradient-based optimization:

.. code-block:: python

   import torch
   from spflow.modules import Sum
   from spflow.modules.leaf import Normal
   from spflow.meta import Scope
   from spflow.learn import train_gradient_descent
   from spflow import log_likelihood

   # Create model
   scope = Scope([0, 1])
   leaf_layer = Normal(scope=scope, out_channels=4)
   model = Sum(inputs=leaf_layer, out_channels=1)

   # Generate training data
   torch.manual_seed(42)
   train_data = torch.randn(1000, 2)

   # Train the model
   trained_model = train_gradient_descent(
       model,
       train_data,
       learning_rate=0.01,
       num_epochs=50
   )

   # Evaluate
   ll = log_likelihood(trained_model, train_data)
   print(f"Average log-likelihood: {ll.mean():.4f}")

Understanding the Output
-------------------------

**Shape notation:** SPFlow uses tensor shapes throughout. Understanding the output shapes:

- ``[batch_size, num_features]``: Input data shape
- ``[batch_size, depth, channels]``: Log-likelihood output shape
- ``depth``: Position in the hierarchical structure
- ``channels``: Number of parallel mixture components

**Dispatched functions:** Key functions like ``log_likelihood``, ``sample``, and ``em`` work polymorphically across different module types using the plum-dispatch library.

Next Steps
----------

- Explore the :doc:`tutorials/index` for in-depth examples
- Read the :doc:`architecture` guide to understand SPFlow's design
- Check the :doc:`api/index` for detailed API documentation
- Learn about :doc:`api/learning` algorithms for training SPNs
