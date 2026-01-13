import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

from spflow.interfaces.sklearn import SPFlowClassifier


class _DummyClassifier:
    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([data[:, 0], -data[:, 0]], dim=1)
        return torch.softmax(logits, dim=1)


def test_predict_proba_shape():
    X = np.random.randn(10, 3).astype(np.float32)
    y = np.random.randint(0, 2, size=(10,))

    clf = SPFlowClassifier(model=_DummyClassifier(), dtype="float32")
    clf.fit(X, y)
    probs = clf.predict_proba(X[:4])
    assert probs.shape == (4, 2)


def test_predict_matches_argmax_proba():
    X = np.random.randn(8, 1).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    clf = SPFlowClassifier(model=_DummyClassifier(), dtype="float32")
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    pred = clf.predict(X)
    np.testing.assert_array_equal(pred, np.argmax(probs, axis=1))
