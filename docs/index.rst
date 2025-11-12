SPFlow Documentation
====================

**SPFlow** is a library for **Probabilistic Circuits (PCs)** built on PyTorch, enabling fast and flexible inference and learning for probabilistic graphical models.

Probabilistic Circuits (also known as Sum-Product Networks) are deep probabilistic models that support efficient exact inference. SPFlow provides a PyTorch-based implementation with automatic differentiation, GPU acceleration, and a clean, modular API.

.. grid:: 2

   .. grid-item-card:: 🚀 Getting Started
      :link: installation
      :link-type: doc

      Installation instructions and quick start guide to get up and running with SPFlow.

   .. grid-item-card:: 📚 Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step tutorials and examples to learn SPFlow fundamentals.

.. grid:: 2

   .. grid-item-card:: 🏗️ Architecture
      :link: architecture
      :link-type: doc

      Understanding SPFlow's design: dispatch system, module hierarchy, and patterns.

   .. grid-item-card:: 📖 API Reference
      :link: api/index
      :link-type: doc

      Complete API documentation for all modules, classes, and functions.

Key Features
------------

- **PyTorch-based:** Built on PyTorch for GPU acceleration and automatic differentiation
- **Modular design:** Clean abstraction with Sum, Product, and Leaf modules
- **Rich distributions:** Support for 13+ probability distributions (both unconditional and conditional)
- **Learning algorithms:** Structure learning (LearnSPN), gradient descent, and EM
- **Dispatch system:** Flexible polymorphic dispatch using plum-dispatch
- **Type-safe:** Comprehensive type hints throughout the codebase

Quick Example
-------------

.. code-block:: python

   import torch
   from spflow.modules import Sum, Product
   from spflow.modules.leaf import Normal
   from spflow import log_likelihood, sample

   # Create a simple SPN
   leaf1 = Normal(scope=[0], mean=0.0, std=1.0)
   leaf2 = Normal(scope=[0], mean=2.0, std=1.0)
   sum_node = Sum(children=[leaf1, leaf2], weights=[0.5, 0.5])

   # Compute log-likelihood
   data = torch.tensor([[1.5]])
   ll = log_likelihood(sum_node, data)

   # Generate samples
   samples = sample(sum_node, num_samples=100)

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   architecture
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
