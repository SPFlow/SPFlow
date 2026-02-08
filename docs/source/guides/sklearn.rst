=========================
Using SPFlow with sklearn
=========================

SPFlow provides optional scikit-learn compatible wrappers in :mod:`spflow.interfaces.sklearn`.

Installation
============

Install with the sklearn extra::

    pip install spflow[sklearn]

Density Estimation
==================

.. code-block:: python

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from spflow.interfaces.sklearn import SPFlowDensityEstimator

    X = np.random.randn(500, 2).astype(np.float32)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pc", SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32")),
        ]
    )
    pipe.fit(X)

    logp = pipe.named_steps["pc"].score_samples(pipe.named_steps["scaler"].transform(X[:5]))
    samples = pipe.named_steps["pc"].sample(10, random_state=0)

Differentiable Sampling (``rsample``)
=====================================

Use ``rsample`` when you need gradients through sampling. For autograd, set
``return_tensor=True`` (default output is NumPy, which detaches gradients)::

    import torch
    from spflow.interfaces.sklearn import SPFlowDensityEstimator

    est = SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32")
    est.fit(X)

    out = est.rsample(
        n_samples=16,
        method="simple",
        tau=1.0,
        hard=True,
        return_tensor=True,
    )
    loss = out.mean()
    loss.backward()

``method``, ``tau``, and ``hard`` control the differentiable categorical routing behavior.
For background, see:
`Elevating Perceptual Sample Quality in Probabilistic Circuits through Differentiable Sampling <https://proceedings.mlr.press/v181/lang22a/lang22a.pdf>`_.

Classifier Wrapper
==================

If you already have an SPFlow model that implements ``predict_proba(torch.Tensor)``,
wrap it as a scikit-learn classifier:

.. code-block:: python

    import numpy as np
    from spflow.interfaces.sklearn import SPFlowClassifier

    # model = ...  # any SPFlow classifier providing predict_proba(torch.Tensor)
    # X, y = ...
    # clf = SPFlowClassifier(model=model, dtype="float32").fit(X, y)
    # y_proba = clf.predict_proba(X)
