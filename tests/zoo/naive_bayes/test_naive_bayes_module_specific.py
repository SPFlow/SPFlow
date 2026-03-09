from __future__ import annotations

import importlib

import pytest
import torch

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.learn.expectation_maximization import expectation_maximization
from spflow.learn.gradient_descent import train_gradient_descent
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.utils.cache import Cache
from spflow.zoo.naive_bayes import NaiveBayes
from tests.utils.leaves import make_leaf


def test_naive_bayes_is_exported_from_subpackage():
    module = importlib.import_module("spflow.zoo.naive_bayes")

    assert module.NaiveBayes is NaiveBayes


def test_root_zoo_package_exports_naive_bayes():
    module = importlib.import_module("spflow.zoo")

    assert module.NaiveBayes is NaiveBayes


def test_density_log_likelihood_matches_manual_product():
    leaf = Bernoulli(
        scope=Scope([0, 1]),
        probs=torch.tensor([[[0.25]], [[0.75]]]),
    )
    model = NaiveBayes(leaf)
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    ll = model.log_likelihood(data)
    expected = torch.log(torch.tensor([[(1 - 0.25) * 0.75], [0.25 * (1 - 0.75)]], dtype=ll.dtype))

    assert ll.shape == (2, 1, 1, 1)
    torch.testing.assert_close(ll.squeeze(-1).squeeze(-1), expected, rtol=1e-6, atol=1e-6)


def test_density_mle_updates_leaf_parameters():
    leaf = Bernoulli(scope=Scope([0]), probs=torch.tensor([[[0.5]]]))
    model = NaiveBayes(leaf)
    data = torch.tensor([[0.0], [1.0], [1.0], [1.0]])

    model.maximum_likelihood_estimation(data)

    torch.testing.assert_close(leaf.probs.squeeze(), torch.tensor(0.75), rtol=1e-6, atol=1e-6)


def test_classifier_log_posterior_matches_manual_bayes_rule():
    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.8], [0.2]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, class_prior=torch.tensor([0.75, 0.25]))
    data = torch.tensor([[1.0], [0.0]])

    log_post = model.log_posterior(data)
    probs = model.predict_proba(data)

    expected = torch.tensor([[12.0 / 13.0, 1.0 / 13.0], [3.0 / 7.0, 4.0 / 7.0]], dtype=probs.dtype)
    torch.testing.assert_close(probs, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(log_post.exp(), expected, rtol=1e-6, atol=1e-6)


def test_classifier_log_likelihood_matches_manual_marginal():
    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.8], [0.2]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, class_prior=torch.tensor([0.75, 0.25]))
    data = torch.tensor([[1.0], [0.0]])

    ll = model.log_likelihood(data)
    expected = torch.log(torch.tensor([[0.65], [0.35]], dtype=ll.dtype))

    torch.testing.assert_close(ll.squeeze(-1).squeeze(-1), expected, rtol=1e-6, atol=1e-6)


def test_log_posterior_raises_for_single_class():
    model = NaiveBayes(Bernoulli(scope=Scope([0]), probs=torch.tensor([[[0.5]]])))
    with pytest.raises(UnsupportedOperationError):
        model.log_posterior(torch.tensor([[0.0]]))


def test_classifier_supervised_mle_updates_leaf_channels_and_learnable_prior():
    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.5], [0.5]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, learnable_prior=True)
    data = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    targets = torch.tensor([0, 0, 0, 1], dtype=torch.long)

    model.maximum_likelihood_estimation(data, targets=targets)

    assert leaf.probs[0, 0, 0] < leaf.probs[0, 1, 0]
    torch.testing.assert_close(
        model.root_node.weights.squeeze(0).squeeze(-1).squeeze(-1),
        torch.tensor([0.75, 0.25], dtype=model.root_node.weights.dtype),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fixed_prior_stays_frozen_during_supervised_mle():
    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.5], [0.5]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, class_prior=torch.tensor([0.9, 0.1]))
    before = model.root_node.weights.detach().clone()

    model.maximum_likelihood_estimation(
        torch.tensor([[0.0], [1.0], [1.0], [1.0]]),
        targets=torch.tensor([0, 1, 1, 1], dtype=torch.long),
    )

    torch.testing.assert_close(model.root_node.weights, before, rtol=0.0, atol=0.0)


def test_supervised_mle_updates_all_leaves_in_multi_leaf_model():
    left = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.5], [0.5]]]),
    )
    right = Bernoulli(
        scope=Scope([1]),
        probs=torch.tensor([[[0.5], [0.5]]]),
    )
    model = NaiveBayes([left, right], num_classes=2, learnable_prior=True)
    data = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    model.maximum_likelihood_estimation(data, targets=targets)

    assert left.probs[0, 0, 0] < left.probs[0, 1, 0]
    assert right.probs[0, 0, 0] > right.probs[0, 1, 0]


def test_expectation_maximization_bypasses_frozen_prior_node(monkeypatch):
    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.5], [0.5]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, class_prior=torch.tensor([0.5, 0.5]))
    called = {"root": False, "conditional": False}

    def fake_root(data, bias_correction=True, *, cache):
        del data, bias_correction, cache
        called["root"] = True

    def fake_conditional(data, bias_correction=True, *, cache):
        del data, bias_correction, cache
        called["conditional"] = True

    monkeypatch.setattr(model.root_node, "_expectation_maximization_step", fake_root)
    monkeypatch.setattr(model.conditional_root, "_expectation_maximization_step", fake_conditional)

    model._expectation_maximization_step(torch.tensor([[0.0]]), cache=Cache())

    assert called["conditional"] is True
    assert called["root"] is False


def test_expectation_maximization_keeps_fixed_prior_unchanged():
    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.6], [0.4]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, class_prior=torch.tensor([0.8, 0.2]))
    before = model.root_node.weights.detach().clone()
    data = torch.tensor([[0.0], [0.0], [1.0], [1.0], [1.0]])

    expectation_maximization(model, data, max_steps=1)

    torch.testing.assert_close(model.root_node.weights, before, rtol=0.0, atol=0.0)


def test_classifier_sampling_returns_expected_shape():
    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.9], [0.1]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, class_prior=torch.tensor([0.6, 0.4]))
    samples = model.sample(num_samples=5)

    assert samples.shape == (5, 1)
    assert torch.isfinite(samples).all()


def test_prior_trainability_flag_controls_root_logits_grad():
    frozen = NaiveBayes(
        Bernoulli(scope=Scope([0]), probs=torch.tensor([[[0.5], [0.5]]])),
        num_classes=2,
        class_prior=torch.tensor([0.5, 0.5]),
    )
    learnable = NaiveBayes(
        Bernoulli(scope=Scope([0]), probs=torch.tensor([[[0.5], [0.5]]])),
        num_classes=2,
        learnable_prior=True,
    )

    assert frozen.root_node.logits.requires_grad is False
    assert learnable.root_node.logits.requires_grad is True


def test_invalid_constructor_and_fit_inputs():
    with pytest.raises(InvalidParameterError):
        NaiveBayes(
            make_leaf(Bernoulli, out_features=1, out_channels=2, num_repetitions=2),
            num_classes=2,
        )

    with pytest.raises(InvalidParameterError):
        NaiveBayes(make_leaf(Bernoulli, out_features=1, out_channels=1), num_classes=2)

    with pytest.raises(InvalidParameterError):
        NaiveBayes(make_leaf(Bernoulli, out_features=1, out_channels=1), class_prior=[0.5, 0.5])

    model = NaiveBayes(make_leaf(Bernoulli, out_features=1, out_channels=2), num_classes=2)
    with pytest.raises(InvalidParameterError):
        model.maximum_likelihood_estimation(torch.tensor([[0.0]]))


@pytest.mark.parametrize(
    "invalid_targets",
    [
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0.0, 1.0]),
        torch.tensor([True, False]),
        torch.tensor([-1, 1], dtype=torch.long),
        torch.tensor([0, 2], dtype=torch.long),
    ],
)
def test_supervised_mle_rejects_invalid_targets(invalid_targets: torch.Tensor):
    model = NaiveBayes(make_leaf(Bernoulli, out_features=1, out_channels=2), num_classes=2)
    data = torch.tensor([[0.0], [1.0]])

    with pytest.raises(InvalidParameterError):
        model.maximum_likelihood_estimation(data, targets=invalid_targets)


def test_classifier_trains_with_gradient_descent_classification_mode():
    leaf = Bernoulli(
        scope=Scope([0]),
        probs=torch.tensor([[[0.4], [0.6]]]),
    )
    model = NaiveBayes(leaf, num_classes=2, learnable_prior=True)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[0.0], [0.0], [1.0], [1.0]]),
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    train_gradient_descent(model, dataloader, epochs=1, is_classification=True, lr=0.05)

    assert model.root_node.logits.grad is not None
