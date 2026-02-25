import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

from spflow.interfaces.sklearn import SPFlowClassifier


class _DummyClassifier:
    device = "cpu"

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([data[:, 0], -data[:, 0]], dim=1)
        return torch.softmax(logits, dim=1)


class _DummyClassifierNoDevice:
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
    # Classifier API contract: predict delegates to argmax over predict_proba.
    np.testing.assert_array_equal(pred, np.argmax(probs, axis=1))


def test_predict_proba_device_override_and_model_device_paths():
    X = np.random.randn(6, 2).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    clf_model_device = SPFlowClassifier(model=_DummyClassifier(), dtype="float64")
    clf_model_device.fit(X, y)
    probs_model_device = clf_model_device.predict_proba(X[:2])
    assert probs_model_device.shape == (2, 2)

    clf_override = SPFlowClassifier(model=_DummyClassifier(), device="cpu", dtype="float32")
    clf_override.fit(X, y)
    probs_override = clf_override.predict_proba(X[:2])
    assert probs_override.shape == (2, 2)


def test_predict_proba_uses_cpu_fallback_when_model_has_no_device():
    X = np.random.randn(4, 2).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    clf = SPFlowClassifier(model=_DummyClassifierNoDevice(), dtype="float32")
    clf.fit(X, y)
    # Models without .device should still run through the CPU fallback path.
    probs = clf.predict_proba(X)
    assert probs.shape == (4, 2)
