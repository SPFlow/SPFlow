Architecture Guide
==================

This guide explains SPFlow's design principles, key architectural patterns, and system organization to help you understand and extend the library.

Overview
--------

SPFlow is built on three core architectural pillars:

1. **Module Hierarchy**: A clean inheritance structure for building probabilistic circuits
2. **Dispatch System**: Polymorphic function dispatch for extensible operations
3. **Metadata System**: Scope tracking and feature context management

Module Hierarchy
----------------

All SPFlow components inherit from the base ``Module`` class, which extends PyTorch's ``nn.Module``.

Base Module
^^^^^^^^^^^

The ``Module`` class (``spflow.modules.module.Module``) is the foundation of all SPFlow components:

.. code-block:: python

   from spflow.modules import Module

   class Module(nn.Module, ABC):
       """Abstract base class for all SPFlow modules."""

       def __init__(self, scope: Scope, ...):
           super().__init__()
           self._scope = scope
           # ... other initialization

       @property
       def scope(self) -> Scope:
           """Get the scope (features) this module operates on."""
           return self._scope

Key properties of every module:

- **scope**: The set of feature indices the module operates on
- **out_channels**: Number of parallel output channels
- **out_features**: Number of output features

Inner Nodes
^^^^^^^^^^^

Inner nodes combine children modules to build hierarchical structures:

**Sum Nodes** (``spflow.modules.Sum``):
   - Represent weighted mixtures of child distributions
   - Learn mixture weights (parameters)
   - Scope = union of children scopes

   .. code-block:: python

      from spflow.modules import Sum
      sum_node = Sum(inputs=[child1, child2], out_channels=2)

**Product Nodes** (``spflow.modules.Product``):
   - Represent factorizations (independence assumptions)
   - No learnable parameters
   - Scope = union of children scopes (should be disjoint)

   .. code-block:: python

      from spflow.modules import Product
      product_node = Product(inputs=[child1, child2])

Leaf Nodes
^^^^^^^^^^

Leaf nodes are probability distributions over input features:

**Base Leaf** (``spflow.modules.leaf.LeafModule``):
   - Abstract base for all leaf distributions
   - Wraps ``spflow.distributions.Distribution`` objects
   - Provides parameter learning via EM or gradient descent

**Available Distributions**:
   - Continuous: Normal, LogNormal, Gamma, Exponential, Uniform
   - Discrete: Bernoulli, Binomial, Categorical, Geometric, Poisson, Hypergeometric, NegativeBinomial

   .. code-block:: python

      from spflow.modules.leaf import Normal, Categorical

      # Continuous distribution
      normal_leaf = Normal(scope=[0, 1], out_channels=4)

      # Discrete distribution
      categorical_leaf = Categorical(scope=[2], out_channels=4, K=10)

**Conditional Leaves**:
   SPFlow supports conditional distributions where parameters depend on parent values:

   .. code-block:: python

      from spflow.modules.leaf import CondNormal

      cond_normal = CondNormal(
          scope=[0],
          parent_scope=[1, 2],
          out_channels=4
      )

The Dispatch System
-------------------

SPFlow uses `plum-dispatch <https://github.com/beartype/plum>`_ for polymorphic function dispatch, enabling clean extensibility.

Core Dispatched Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Key operations are defined as dispatched functions that work across all module types:

.. code-block:: python

   from spflow import log_likelihood, sample, em

   # Works with any module type
   ll = log_likelihood(model, data)
   samples = sample(model, num_samples=100)
   model = em(model, data, num_iterations=10)

**Main Dispatched Functions**:

- ``log_likelihood(module, data)``: Compute log P(data | module)
- ``sample(module, evidence, ctx)``: Generate samples
- ``sample_with_evidence(module, evidence, ctx)``: Conditional sampling
- ``em(module, data)``: Expectation-maximization step
- ``maximum_likelihood_estimation(module, data)``: MLE parameter update
- ``marginalize(module, scope)``: Marginalize out variables

How Dispatch Works
^^^^^^^^^^^^^^^^^^

Each module type implements specific versions of dispatched functions:

.. code-block:: python

   from plum import dispatch
   from spflow.modules import Sum, Module

   @dispatch
   def log_likelihood(module: Sum, data: torch.Tensor) -> torch.Tensor:
       """Log-likelihood for Sum nodes."""
       # Compute weighted log-sum-exp over children
       child_ll = log_likelihood(module.children, data)
       weights = torch.log(module.weights)
       return torch.logsumexp(child_ll + weights, dim=-1)

   @dispatch
   def log_likelihood(module: Normal, data: torch.Tensor) -> torch.Tensor:
       """Log-likelihood for Normal leaves."""
       # Use distribution's log_prob
       return module.distribution.log_prob(data)

This allows you to add new module types by simply implementing the dispatched functions for that type.

Metadata System
---------------

Scope
^^^^^

The ``Scope`` class (``spflow.meta.data.Scope``) tracks which features a module operates on:

.. code-block:: python

   from spflow.meta import Scope

   # Scope for features 0, 1, 2
   scope = Scope([0, 1, 2])

   # Check scope membership
   assert 1 in scope
   assert scope.num_features == 3

   # Scope operations
   scope1 = Scope([0, 1])
   scope2 = Scope([2, 3])
   union = scope1 | scope2  # Scope([0, 1, 2, 3])
   intersection = scope1 & scope2  # Scope([])

Scopes enable:
- Automatic validation of module connections
- Tracking of feature dependencies
- Marginalization operations

FeatureContext
^^^^^^^^^^^^^^

The ``FeatureContext`` class (``spflow.meta.data.FeatureContext``) stores metadata about features:

.. code-block:: python

   from spflow.meta import FeatureContext, FeatureTypes

   # Define feature types
   ctx = FeatureContext([
       FeatureTypes.CONTINUOUS,  # Feature 0: continuous
       FeatureTypes.DISCRETE,    # Feature 1: discrete
       FeatureTypes.CONTINUOUS,  # Feature 2: continuous
   ])

   # Query feature information
   assert ctx.is_continuous(0)
   assert ctx.is_discrete(1)

Dispatch Context
^^^^^^^^^^^^^^^^

The ``DispatchContext`` (``spflow.meta.dispatch.DispatchContext``) passes runtime information through dispatched functions:

.. code-block:: python

   from spflow.meta import DispatchContext

   ctx = DispatchContext(
       memoize=True,  # Enable memoization for repeated computations
       gradient_tracking=True,  # Track gradients
   )

   ll = log_likelihood(model, data, ctx=ctx)

Learning Algorithms
-------------------

SPFlow provides multiple learning algorithms for different scenarios:

Structure Learning
^^^^^^^^^^^^^^^^^^

``learn_spn`` (``spflow.learn.learn_spn.learn_spn``) automatically learns SPN structure:

.. code-block:: python

   from spflow.learn import learn_spn
   from spflow.modules.leaf import Normal

   model = learn_spn(
       data,
       leaf_modules=Normal(scope=scope, out_channels=4),
       out_channels=1,
       min_instances_slice=100  # Minimum instances for splitting
   )

**Algorithm**: LearnSPN uses recursive data partitioning:
1. Check for independence → create Product node
2. Cluster data → create Sum node
3. Base case → create Leaf node
4. Recurse on partitions

Parameter Learning
^^^^^^^^^^^^^^^^^^

**Gradient Descent** (``spflow.learn.gradient_descent``):
   - Uses PyTorch's autograd
   - Maximizes log-likelihood via gradient ascent
   - Supports standard optimizers (Adam, SGD, etc.)

   .. code-block:: python

      from spflow.learn import train_gradient_descent

      trained_model = train_gradient_descent(
          model, data,
          learning_rate=0.01,
          num_epochs=100
      )

**Expectation-Maximization** (``spflow.learn.expectation_maximization``):
   - Closed-form updates for certain distributions
   - No gradient computation required
   - Often faster for simple models

   .. code-block:: python

      from spflow.learn import expectation_maximization

      trained_model = expectation_maximization(
          model, data,
          num_iterations=20
      )

RAT-SPN Architecture
--------------------

RAT-SPNs (``spflow.modules.rat.RatSPN``) are efficient tensorized architectures:

**Key Features**:
- Random variable partitioning at each layer
- Tensorized sum and product operations
- Efficient GPU computation
- Scalable to high-dimensional data

.. code-block:: python

   from spflow.modules.rat import RatSPN
   from spflow.modules.leaf import Normal

   model = RatSPN(
       leaf_modules=[Normal(scope=scope, out_channels=4)],
       n_root_nodes=1,
       n_region_nodes=8,
       num_repetitions=2,
       depth=3,
       outer_product=False
   )

**Architecture**:
- Layer 0: Leaf distributions
- Layers 1..depth-1: Alternating sum/product mixing layers
- Layer depth: Root sum node

Visualization
-------------

SPFlow provides visualization tools for understanding model structure:

.. code-block:: python

   from spflow.utils.visualization import visualize_module

   visualize_module(
       model,
       output_path="model_structure",
       show_scope=True,
       show_shape=True,
       show_params=True,
       format="png"  # or "svg", "pdf"
   )

This generates a GraphViz visualization showing:
- Node types and connections
- Scopes for each node
- Tensor shapes
- Learnable parameters

Design Patterns
---------------

Channel-based Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^

SPFlow uses channels to represent multiple copies of distributions:

.. code-block:: python

   # 4 parallel Gaussian distributions
   leaf = Normal(scope=[0], out_channels=4)

Channels enable:
- Efficient batch computation
- Multiple mixture components
- Hierarchical combinations

Lazy Initialization
^^^^^^^^^^^^^^^^^^^

Modules lazily initialize parameters on first forward pass, allowing flexible construction:

.. code-block:: python

   leaf = Normal(scope=[0], out_channels=4)
   # Parameters created on first log_likelihood call
   ll = log_likelihood(leaf, data)

Immutable Structures
^^^^^^^^^^^^^^^^^^^^

Once built, SPN structures are typically immutable. Learning modifies parameters, not structure (except for structure learning algorithms).

Extending SPFlow
----------------

To add a new distribution:

1. Create distribution class in ``spflow/distributions/``
2. Create leaf module in ``spflow/modules/leaf/``
3. Implement dispatched functions (``log_likelihood``, ``sample``, etc.)
4. Add unit tests

Example skeleton:

.. code-block:: python

   # spflow/distributions/my_distribution.py
   from spflow.distributions import Distribution

   class MyDistribution(Distribution):
       def log_prob(self, value):
           # Implement log probability
           pass

   # spflow/modules/leaf/my_leaf.py
   from spflow.modules.leaf import LeafModule

   class MyLeaf(LeafModule):
       def __init__(self, scope, out_channels, **params):
           dist = MyDistribution(**params)
           super().__init__(scope, out_channels, dist)

   # Implement dispatched functions
   @dispatch
   def log_likelihood(module: MyLeaf, data: torch.Tensor):
       return module.distribution.log_prob(data)

Summary
-------

SPFlow's architecture provides:

- **Clean abstractions**: Module hierarchy with clear responsibilities
- **Extensibility**: Dispatch system for adding new operations
- **Type safety**: Comprehensive type hints throughout
- **Performance**: PyTorch backend with GPU acceleration
- **Flexibility**: Multiple learning algorithms and model architectures

This design enables building complex probabilistic models while maintaining code clarity and extensibility.
