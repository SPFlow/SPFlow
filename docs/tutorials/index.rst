Tutorials
=========

These comprehensive tutorials walk you through SPFlow's features with interactive examples.

.. note::
   These tutorials are Jupyter notebooks that can be downloaded and run locally. The notebooks are also available in the `guides/ <https://github.com/SPFlow/SPFlow/tree/develop/guides>`_ directory of the repository.

Available Tutorials
-------------------

.. toctree::
   :maxdepth: 1

   user_guide
   user_guide2

Tutorial 1: User Guide (Basic)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An introduction to SPFlow covering:

- Basic model construction
- Sum and Product nodes
- Leaf distributions
- Computing log-likelihood
- Sampling from models

.. raw:: html

   <p><a href="user_guide.html">View Tutorial 1 →</a></p>

Tutorial 2: User Guide (Advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Advanced SPFlow features including:

- RAT-SPN architectures
- Structure learning with LearnSPN
- Parameter learning (gradient descent and EM)
- Conditional distributions
- Missing data handling
- Visualization

.. raw:: html

   <p><a href="user_guide2.html">View Tutorial 2 →</a></p>

Running Tutorials Locally
--------------------------

To run these tutorials on your machine:

1. Install SPFlow with Jupyter dependencies:

   .. code-block:: bash

      pip install spflow jupyter matplotlib

2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/SPFlow/SPFlow.git
      cd SPFlow/guides

3. Launch Jupyter:

   .. code-block:: bash

      jupyter notebook

4. Open ``user_guide.ipynb`` or ``user_guide2.ipynb``

What's Next?
------------

After completing these tutorials:

- Explore the :doc:`../examples/index` gallery for specific use cases
- Read the :doc:`../architecture` guide for design details
- Check the :doc:`../api/index` for comprehensive API documentation
