.. _spflow-api:

SPFlow API
==========

.. currentmodule:: spn

Structure
---------

.. autosummary::
  :toctree: generated/

  spn.structure.Base.Context
  spn.structure.Base.Sum
  spn.structure.Base.Product
  spn.structure.Base.assign_ids
  spn.structure.Base.rebuild_scopes_bottom_up
  spn.structure.StatisticalTypes.MetaType
  spn.structure.leaves.cltree.CLTree.create_cltree_leaf
  spn.structure.leaves.parametric.Parametric.Bernoulli
  spn.structure.leaves.parametric.Parametric.Categorical
  spn.structure.leaves.parametric.Parametric.Gaussian

Learning
--------

.. autosummary::
  :toctree: generated/

  spn.algorithms.LearningWrappers.learn_classifier
  spn.algorithms.LearningWrappers.learn_cnet
  spn.algorithms.LearningWrappers.learn_mspn
  spn.algorithms.LearningWrappers.learn_parametric

Inference
---------

.. autosummary::
  :toctree: generated/

  spn.algorithms.Inference.log_likelihood
  spn.algorithms.Marginalization.marginalize
  spn.algorithms.MPE.mpe

Utility Methods
---------------

These generally relate to visualization, statistics, and whether the structure
is valid.

.. autosummary::
  :toctree: generated/

  spn.io.Graphics.draw_spn
  spn.io.Graphics.plot_spn
  spn.algorithms.Statistics.get_structure_stats
  spn.algorithms.Validity.is_valid

Datasets
--------

The following data sets are included here:

.. autosummary::
  :toctree: generated/

  spn.data.datasets
