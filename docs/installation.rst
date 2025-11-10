Installation
============

Requirements
------------

SPFlow requires:

- **Python:** 3.10 or higher
- **PyTorch:** 2.0.1 or higher
- **NumPy:** 1.26.4 or higher
- Additional dependencies (automatically installed): scipy, scikit-learn, plum-dispatch, pydot

.. note::
   SPFlow 1.0.0 is a complete rewrite using PyTorch and has not yet been officially released. The pre-v1.0.0 version is still available on PyPI (``spflow==0.0.46``) and in the ``legacy`` branch of the repository.

Installation Methods
--------------------

Option 1: PyPI Installation (Recommended for Users)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to install SPFlow is via pip:

.. code-block:: bash

   pip install spflow

This will install SPFlow and all its required dependencies.

Option 2: Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For developers who want to contribute or work with the latest development version:

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/SPFlow/SPFlow.git
   cd SPFlow

2. Install with development dependencies:

.. code-block:: bash

   uv sync --extra dev

This installs SPFlow in development mode with all development tools (testing, linting, documentation, etc.).

Verifying Installation
----------------------

To verify that SPFlow is installed correctly, run the following in Python:

.. code-block:: python

   import spflow
   print(f"SPFlow version: {spflow.__version__}")

   # Try importing key modules
   from spflow.modules import Sum, Product
   from spflow.modules.leaf import Normal
   from spflow import log_likelihood, sample

   print("SPFlow installed successfully!")

If this runs without errors, SPFlow is installed and ready to use.

GPU Support
-----------

SPFlow is built on PyTorch and automatically uses GPU acceleration when available. To enable GPU support:

1. Install PyTorch with CUDA support according to your system configuration. See the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

2. Verify GPU availability:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

SPFlow modules (tensors and models) can be moved to GPU using standard PyTorch methods:

.. code-block:: python

   import torch
   from spflow.modules import Sum
   from spflow.modules.leaf import Normal

   # Create model
   model = Sum(inputs=Normal(scope=[0, 1], out_channels=4), out_channels=1)

   # Move to GPU
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   data = torch.randn(32, 2).to(device)

Troubleshooting
---------------

**Import errors:**
   Make sure all dependencies are installed. If you encounter issues with specific packages, try updating them:

   .. code-block:: bash

      pip install --upgrade torch numpy scipy scikit-learn

**PyTorch version conflicts:**
   SPFlow requires PyTorch >= 2.0.1. If you have an older version, upgrade it:

   .. code-block:: bash

      pip install --upgrade torch

**Development dependencies not found:**
   If you're developing SPFlow and encounter missing tools, ensure you installed with the ``dev`` extra:

   .. code-block:: bash

      uv sync --extra dev

Next Steps
----------

- Continue to the :doc:`quickstart` guide for your first SPFlow model
- Check out the :doc:`tutorials/index` for detailed examples
- Explore the :doc:`api/index` for comprehensive API documentation
