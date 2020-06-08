Development
===========

SPFlow is meant to be built on and extended. This short guide will help you get
started on extending it further: pointing out testing, documentation, and some
projects using SPFlow.

Testing
-------

Testing is done with pytest, and there are a few additional libraries which
may be needed during development.

Assuming you are on Ubuntu:

.. code-block:: bash

  sudo apt-get install g++-7

... and some additional Python packages:

.. code-block:: bash

  pip install -r requirements.txt
  pip install tensorflow==1.15.0
  pip install pytest-xdist
  pip install torch
  pip install torchvision
  pip install cppyy

Tests are based on ``pytest``. From the base of the repository:

.. code-block:: bash

  find src/spn/tests/test*.py -print0 | xargs -n 1 -0 py.test

Documentation
-------------

Documentation is built with ``sphinx``, and has its own set of dependencies
for building:

.. code-block:: bash

  cd docs/
  pip install -r requirements.txt

Once these are installed, a local copy of the documentation can be built
using the ``Makefile`` (OSX/Linux) or ``make.bat`` (Windows).

.. code-block::

  make html
  xdg-open build/html/index.html  # Open in default web browser

Projects using SPFlow
---------------------

There are several projects currently using SPFlow for a variety of tasks.
Here we have links to a few of them:

- `DeepNoteBooks <https://github.com/cvoelcker/DeepNotebooks>`_: *DeepNotebooks
  is an automated statistical analysis tool build on top of SPNs.*
- `CryptoSPN <https://github.com/encryptogroup/CryptoSPN>`_: *an extension of
  SPFlow to allow for privacy-preserving SPN inference.*
- `DeepDB <https://github.com/DataManagementLab/deepdb-public>`_: *a data-driven
  learned database component achieving state-of-the-art-performance in
  cardinality estimation and approximate query processing (AQP).*
- `Interpreting-SPNs <https://github.com/ml-research/Interpreting-SPNs>`_:
  *Interpreting Sum-Product Networks via Influence Functions.*
