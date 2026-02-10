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

Sampling Notes
==============

The sklearn density estimator exposes ``sample`` only. Differentiable-sampling APIs were removed.

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
