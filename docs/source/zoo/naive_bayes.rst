===================
Naive Bayes
===================

The Paper Zoo includes a lightweight :class:`spflow.zoo.NaiveBayes` wrapper that builds a Naive Bayes model from standard SPFlow leaf modules.

Overview
--------

``NaiveBayes`` supports two common modes:

- **Density estimation** with a single output channel, represented as ``leaf -> Product``.
- **Classification** with one output channel per class and a learned or fixed class prior, represented as ``leaf -> Product -> Sum``.

This makes it easy to reuse existing SPFlow leaves while keeping the familiar Naive Bayes factorization.

Minimal Example
---------------

.. code-block:: python

    import torch
    from spflow.meta.data.scope import Scope
    from spflow.modules.leaves import Bernoulli
    from spflow.zoo import NaiveBayes

    leaf = Bernoulli(
        scope=Scope([0, 1]),
        probs=torch.tensor([[[0.25]], [[0.75]]]),
    )
    model = NaiveBayes(leaf)

    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    ll = model.log_likelihood(x)

Classifier Example
------------------

.. code-block:: python

    import torch
    from spflow.meta.data.scope import Scope
    from spflow.modules.leaves import Bernoulli
    from spflow.zoo import NaiveBayes

    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.8], [0.2]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, class_prior=torch.tensor([0.75, 0.25]))

    x = torch.tensor([[1.0], [0.0]])
    probs = model.predict_proba(x)

API Reference
-------------

.. autoclass:: spflow.zoo.NaiveBayes
   :members:
   :show-inheritance:
