Interfaces
==========

Abstract base classes defining standard interfaces for SPFlow modules.

Classifier
----------

Abstract base class for modules that support classification.

.. autoclass:: spflow.interfaces.classifier.Classifier
   :members:
   :undoc-members:
   :show-inheritance:

sklearn Wrappers
----------------

scikit-learn compatible wrappers for density estimation and classification.

.. autoclass:: spflow.interfaces.sklearn.SPFlowDensityEstimator
   :members:

.. autoclass:: spflow.interfaces.sklearn.SPFlowClassifier
   :members:
