==========================
Frequently Asked Questions
==========================

This page answers common questions about SPFlow. For deeper explanations, see :doc:`Concepts <concepts>`.
For end-to-end tutorials, see the :doc:`User Guide <guides/user_guide>`.

----


General Questions
=================

What is SPFlow?
---------------

SPFlow is a Python library for building and learning **Probabilistic Circuits (PCs)**, including Sum-Product Networks (SPNs). These are deep generative and discriminative models that enable tractable (polynomial-time) probabilistic inference while maintaining expressive power.

SPFlow is built on `PyTorch <https://pytorch.org/>`_, providing GPU acceleration and seamless integration with modern deep learning workflows.

What version of Python is required?
-----------------------------------

SPFlow requires **Python 3.10+** and PyTorch 2.0+.


How do I install SPFlow?
------------------------

See the :doc:`Getting Started <getting_started>` guide for installation instructions. Quick summary::

    pip install spflow

----

Architecture & Concepts
=======================

What are the main module types in SPFlow?
-----------------------------------------

SPFlow provides several core module types:

- **Leaves**: Probability distributions at the terminals (Normal, Categorical, Bernoulli, etc.)
- **Products**: Combine independent distributions (Product, OuterProduct, ElementwiseProduct)
- **Sums**: Weighted mixtures of distributions (Sum, ElementwiseSum)
- **Specialized architectures**: RAT-SPN, ConvPc for images

See the :doc:`API Reference <api/index>` for complete documentation.

What is a Scope?
----------------

A **Scope** defines which input variables (features) a module operates on. Scopes are what make sums/products well-defined
and enforce decomposability/compatibility constraints.

See :ref:`concepts-scopes-and-decomposability`.

What are repetitions?
---------------------

**Repetitions** are independent parameterizations of the same structure.
They usually show up as ``R`` in ``model.to_str()`` and are tracked in module shapes.

See :ref:`concepts-shapes-and-dimensions`.

What is the difference between Sum and ElementwiseSum?
------------------------------------------------------

- **Sum**: Computes weighted mixtures over all input channels. Output has ``out_channels`` channels, each being a weighted combination of all input channels.

- **ElementwiseSum**: Sums corresponding channels across multiple input modules element-wise. Requires all inputs to have the same scope and channel count.

What is the difference between Product, OuterProduct, and ElementwiseProduct?
------------------------------------------------------------------------------

- **Product**: Combines inputs by computing products across all features. The inputs must have disjoint scopes.

- **OuterProduct**: Computes the outer product of split inputs. Takes input split into groups and produces all combinations.

- **ElementwiseProduct**: Multiplies corresponding elements across multiple input modules. Requires inputs with compatible shapes.

----

Model Building
==============

How do I create a simple SPN?
-----------------------------

Here's a minimal example::

    import torch
    from spflow.modules.sums import Sum
    from spflow.modules.products import Product
    from spflow.modules.leaves import Normal
    from spflow.meta import Scope

    # Create leaves for 2 features
    scope = Scope([0, 1])
    leaves = Normal(scope=scope, out_channels=4)

    # Stack product and sum layers
    product = Product(inputs=leaves)
    model = Sum(inputs=product, out_channels=1)

    # Use the model
    data = torch.randn(32, 2)
    log_ll = model.log_likelihood(data)

See the :doc:`Getting Started <getting_started>` guide for more examples.

What leaf distributions are available?
--------------------------------------

SPFlow includes many univariate distributions:

**Continuous**: Normal, LogNormal, Exponential, Gamma, Uniform

**Discrete**: Categorical, Bernoulli, Binomial, Poisson, Geometric, NegativeBinomial, Hypergeometric

See :doc:`api/leaves` for complete documentation.

How do I use RAT-SPN?
---------------------

RAT-SPN (Randomized And Tensorized SPN) automatically builds a deep circuit from hyperparameters::

    from spflow.modules.rat import RatSPN
    from spflow.modules.leaves import Normal
    from spflow.meta import Scope

    # Create leaves
    scope = Scope(list(range(64)))
    leaves = Normal(scope=scope, out_channels=4, num_repetitions=2)

    # Build RAT-SPN
    model = RatSPN(
        leaf_modules=[leaves],
        n_root_nodes=1,
        n_region_nodes=8,
        num_repetitions=2,
        depth=3
    )

See :doc:`api/rat_spn` for details.

Does SPFlow have image-specific modules?
-----------------------------------------

Yes! Use the ``ConvPc`` module for image data with spatial structure::

    from spflow.modules.conv import ConvPc
    from spflow.modules.leaves import Binomial
    from spflow.meta import Scope

    # Create leaf layer for 28x28 grayscale images (e.g., MNIST)
    height, width = 28, 28
    scope = Scope(list(range(height * width)))
    leaf = Binomial(scope=scope, total_count=torch.tensor(255), out_channels=8, num_repetitions=2)

    # Build convolutional PC
    model = ConvPc(
        leaf=leaf,
        input_height=height,
        input_width=width,
        depth=3,
        channels=16,
        kernel_size=2,
        num_repetitions=2,
    )

For adapting existing models to image data, use ``ImageWrapper``::

    from spflow.modules.wrapper import ImageWrapper

    # Wrap any SPFlow model for image data
    wrapped = ImageWrapper(model, num_channel=1, height=28, width=28)

    # Now works with 4D tensors: (batch, channels, height, width)
    image_data = torch.randn(32, 1, 28, 28)
    log_ll = wrapped.log_likelihood(image_data)

See :doc:`api/conv` and :doc:`api/wrappers` for complete documentation.

----

Training & Learning
===================

How do I train a model?
-----------------------

SPFlow provides two main training approaches:

**Gradient Descent**::

    from spflow.learn import train_gradient_descent

    train_gradient_descent(
        model,
        train_data,
        epochs=100,
        lr=0.01
    )

**Expectation-Maximization**::

    from spflow.learn import expectation_maximization

    expectation_maximization(model, train_data, epochs=50)

What is the difference between gradient descent and EM?
-------------------------------------------------------

Both methods use gradients in SPFlow's implementation:

- **Gradient Descent**: Standard PyTorch optimization. Suitable for most cases, especially when combined with other neural network components.

- **Expectation-Maximization (EM)**: A specialized algorithm that alternates between computing expected sufficient statistics and updating parameters. This is usually more stable and converges faster than gradient descent.

Choose based on your use case; gradient descent is generally more flexible.

How do I use structure learning?
--------------------------------

Use ``learn_spn`` to automatically learn circuit structure from data::

    from spflow.learn import learn_spn
    from spflow.modules.leaves import Normal
    from spflow.meta import Scope

    scope = Scope(list(range(10)))
    leaves = Normal(scope=scope, out_channels=4)

    model = learn_spn(
        data,
        leaf_modules=leaves,
        out_channels=1,
        min_instances_slice=100
    )

See :doc:`api/learning` for details and the :doc:`User Guide <guides/user_guide>` for end-to-end examples.

----

Inference & Sampling
====================

How do I compute log-likelihood?
--------------------------------

Call the ``log_likelihood`` method on your model::

    log_likelihood = model.log_likelihood(data)
    # Returns tensor of shape [batch_size, ...]

How do I sample from a model?
-----------------------------

Use the ``sample`` method::

    # Generate 100 unconditional samples
    samples = model.sample(num_samples=100)

For conditional sampling with evidence, use ``sample_with_evidence``::

    # Sample some features given others
    evidence = torch.full((10, num_features), float('nan'))
    evidence[:, 0] = 0.5  # Condition on feature 0
    samples = model.sample_with_evidence(evidence=evidence)

What is MPE (Most Probable Explanation) sampling?
-------------------------------------------------

MPE returns the most probable state of the model, useful for generating
clearer outputs and validating training. This is also known as MAP (Maximum
A Posteriori) sampling.

.. note::

   MAP sampling is different from MMAP (Marginal MAP) sampling, which
   marginalizes over some variables while maximizing over others.

::

    # Get the most probable sample
    mpe_sample = model.sample(num_samples=1, is_mpe=True)

MPE can be combined with evidence for conditional MPE::

    evidence = torch.full((1, num_features), float('nan'))
    evidence[0, :10] = observed_values[:10]  # Condition on first 10 features
    conditional_mpe = model.mpe(data=evidence)


How do I handle missing data?
-----------------------------

Use ``torch.nan`` in your data/evidence tensor to indicate missing values::

    # Create data with missing values
    data = torch.randn(100, 5)
    data[0, 2] = float('nan')  # Feature 2 is missing for sample 0
    data[1, 0:2] = float('nan')  # Features 0-1 missing for sample 1

    # Log-likelihood handles missing data automatically
    log_ll = model.log_likelihood(data)

SPFlow will marginalize over missing features when computing likelihoods.

See :ref:`concepts-missing-data-and-evidence` for details and conditional sampling patterns.

----

Visualization & Debugging
=========================

How do I visualize a circuit?
-----------------------------

Use the ``visualize`` function::

    from spflow.utils.visualization import visualize

    visualize(
        model,
        output_path="/tmp/my_circuit",
        format="pdf",
        show_scope=True,
        show_shape=True
    )

Requires `Graphviz <https://graphviz.org/>`_ to be installed on your system.

What output formats are supported?
----------------------------------

The visualization function supports multiple formats via Graphviz:

- **PDF**: ``format="pdf"`` (recommended for papers)
- **SVG**: ``format="svg"`` (scalable, good for web)
- **PNG**: ``format="png"`` (raster image)

How do I print the model structure?
-----------------------------------

Use the ``to_str()`` method for a text representation::

    print(model.to_str())

    # Example output:
    # Sum [D=1, C=1] [weights: (1, 4, 1)] → scope: 0-1
    # └─ Product [D=1, C=4] → scope: 0-1
    #    └─ Normal [D=2, C=4] → scope: 0-1

----


What's the difference between SPFlow v1.x and the legacy version?
-----------------------------------------------------------------

SPFlow v1.0 is a **complete rewrite** using PyTorch as the primary backend. Key differences:

- Modern PyTorch architecture for GPU acceleration
- Significantly improved performance
- Enhanced modular design with composable layers

The pre-v1.0.0 version is still available:

- On PyPI: ``pip install spflow==0.0.46``
- In the ``legacy`` branch of the GitHub repository

Models from the legacy version are **not compatible** with v1.x and need to be rebuilt.

Migration from Legacy
=====================

How do I migrate from SPFlow 0.x to 1.x?
----------------------------------------

SPFlow 1.0 is a complete rewrite. Key changes:

1. **PyTorch-based**: All modules are ``nn.Module`` subclasses
2. **Layered composition**: Build circuits by stacking modules
3. **New API**: Method names and signatures have changed
4. **GPU support**: Native CUDA acceleration

There is no automatic migration path. You will need to:

1. Reinstall: ``pip install --upgrade spflow`` (uninstall legacy first if needed)
2. Rebuild your models using the new API
3. Retrain your models

Are old models compatible with SPFlow v1.x?
-------------------------------------------

**No.** Models saved with SPFlow 0.x cannot be loaded in SPFlow 1.x due to the complete architectural rewrite.

You must rebuild and retrain your models using the new API. See the :doc:`guides/user_guide` for comprehensive examples.
