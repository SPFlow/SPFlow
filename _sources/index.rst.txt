###############################################################
SPFlow: An Easy and Extensible Library for Sum-Product Networks
###############################################################

SPFlow, an open-source Python library providing a simple interface to inference,
learning and manipulation routines for deep and tractable probabilistic models
called Sum-Product Networks (SPNs). The library allows one to quickly create
SPNs both from data and through a domain specific language (DSL). It efficiently
implements several probabilistic inference routines like computing marginals,
conditionals and (approximate) most probable explanations (MPEs) along with
sampling as well as utilities for serializing, plotting and structure statistics
on an SPN.

Furthermore, SPFlow is extremely extensible and customizable, allowing users to
promptly create new inference and learning routines by injecting custom code
into a light-weight functional-oriented API framework.

:ref:`Getting Started <installation-instructions-link>`
-------------------------------------------------------

This walks through the steps of getting SPFlow running. Then check out the
:ref:`Tutorials <basic_examples>` to dive into some examples.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   usage/installation
   usage/quickstart

:ref:`Documentation <spflow-api>`
---------------------------------

This includes API Documentation and some notes on developing with
SPFlow.

.. toctree::
  :maxdepth: 1
  :hidden:
  :caption: Documentation

  documentation/api
  documentation/developing

:ref:`Example Gallery <basic_examples>`
---------------------------------------

A gallery of examples with figures and expected outputs.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   auto_examples/index

Papers SPFlow can reproduce
---------------------------

- Nicola Di Mauro, Antonio Vergari, Teresa M.A. Basile, Floriana Esposito. "Fast and Accurate Density Estimation with Extremely Randomized Cutset Networks". In: ECML/PKDD, 2017. https://doi.org/10.1007/978-3-319-71249-9_13
- Nicola Di Mauro, Antonio Vergari, and Teresa M.A. Basile. "Learning Bayesian Random Cutset Forests". In ISMIS 2015, LNAI 9384, pp. 1-11, Springer, 2015. https://doi.org/10.1007/978-3-319-25252-0_13
- Nicola Di Mauro, Antonio Vergari, and Floriana Esposito. "Learning Accurate Cutset Networks by Exploiting Decomposability". In AI*IA. 2015, LNAI 9336, 1-12, Springer, 2015. https://doi.org/10.1007/978-3-319-24309-2_17
- Antonio Vergari, Nicola Di Mauro, and Floriana Esposito. "Simplifying, Regularizing and Strengthening Sum-Product Network Structure Learning". In ECML/PKDD, LNCS, 343-358, Springer. 2015. https://doi.org/10.1007/978-3-319-23525-7_21

Papers implemented in SPFlow
----------------------------

- Molina, Alejandro, Sriraam Natarajan, and Kristian Kersting. "Poisson Sum-Product Networks: A Deep Architecture for Tractable Multivariate Poisson Distributions." In AAAI, pp. 2357-2363. 2017. https://dl.acm.org/doi/10.5555/3298483.3298579
- Molina, Alejandro, Antonio Vergari, Nicola Di Mauro, Sriraam Natarajan, Floriana Esposito, and Kristian Kersting. "Mixed sum-product networks: A deep architecture for hybrid domains." In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI). 2018. https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16865

Citation
--------

If you find SPFlow useful please cite the `SPFlow arXiv paper <https://arxiv.org/abs/1901.03704>`_ in work:

.. code-block:: bibtex

  @misc{Molina2019SPFlow,
    Author = {Alejandro Molina and Antonio Vergari and Karl Stelzner and Robert Peharz and Pranav Subramani and Nicola Di Mauro and Pascal Poupart and Kristian Kersting},
    Title = {SPFlow: An Easy and Extensible Library for Deep Probabilistic Learning using Sum-Product Networks},
    Year = {2019},
    Eprint = {arXiv:1901.03704},
  }

Authors
-------

- Alejandro Molina - *TU Darmstadt*
- Antonio Vergari - *Max-Planck-Institute*
- Karl Stelzner - *TU Darmstadt*
- Robert Peharz - *University of Cambridge*
- Nicola Di Mauro - *University of Bari Aldo Moro*
- Kristian Kersting - *TU Darmstadt*

Contributors
------------

See also the list of `contributors <https://github.com/SPFlow/SPFlow/contributors>`_ who participated in this project.

- Moritz Kulessa - *TU Darmstadt*
- Claas Voelcker - *TU Darmstadt*
- Simon Roesler - *Karlsruhe Institute of Technology*
- Steven Lang - *TU Darmstadt*
- Xiaoting Shao
- Pranav Subramani - *Wolfram Research*
- Maximilian Gottschalk
- Renato Lui Geh - *University of São Paulo*
- Andy Shih - *Stanford University*
- `Alexander L. Hayes <https://hayesall.com/>`_ - *Indiana University, Bloomington*

License
-------

This project is licensed under the Apache License, Version 2.0 - see the `LICENSE <https://github.com/SPFlow/SPFlow/blob/master/LICENSE.md>`_ file for details

Acknowledgments
---------------

.. image:: https://raw.githubusercontent.com/SPFlow/SPFlow/master/Documentation/acknowledgements/bmbf.png
  :height: 100px
  :alt: Federal Ministry of Education and Research

.. image:: https://raw.githubusercontent.com/SPFlow/SPFlow/master/Documentation/acknowledgements/dfg.jpg
  :height: 100px
  :alt: DFG Deutsche Forschungsgemeinschaft

.. image:: https://raw.githubusercontent.com/SPFlow/SPFlow/master/Documentation/acknowledgements/euc.png
  :height: 100px
  :alt: European Commission

- Parts of SPFlow as well as its motivating research have been supported by the Germany Science Foundation (DFG) - AIPHES, GRK 1994, and CAML, KE 1686/3-1 as part of SPP 1999- and the Federal Ministry of Education and Research (BMBF) - InDaS, 01IS17063B.
- This project received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie Grant Agreement No. 797223 (HYBSPN).
