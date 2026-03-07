"""scikit-learn compatible wrappers for SPFlow models.

These wrappers are optional: SPFlow can be used without scikit-learn installed.
Importing this module does not require scikit-learn, but instantiating the
estimators will.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
import torch
from einops import rearrange, reduce

from spflow.exceptions import (
    InvalidParameterError,
    InvalidTypeError,
    OptionalDependencyError,
    UnsupportedOperationError,
)
from spflow.learn.learn_spn import learn_spn
from spflow.learn.prometheus import learn_prometheus
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module

try:  # pragma: no cover (covered via importorskip tests)
    from sklearn.base import BaseEstimator, ClassifierMixin, DensityMixin
    from sklearn.utils.validation import check_is_fitted

    _SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    BaseEstimator = object  # type: ignore[assignment]
    ClassifierMixin = object  # type: ignore[assignment]
    DensityMixin = object  # type: ignore[assignment]
    check_is_fitted = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise OptionalDependencyError(
            "scikit-learn is required for SPFlow sklearn integration. "
            "Install it with `pip install scikit-learn` (or `pip install spflow[sklearn]`)."
        )


def _as_2d_numpy(x: Any) -> np.ndarray:
    """Convert array-like to a 2D numpy array."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise InvalidParameterError(f"Expected 2D array-like input, got shape {arr.shape}.")
    return arr


def _torch_dtype_from_str(dtype: str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise InvalidParameterError(f"Unknown dtype '{dtype}'. Use 'float32', 'float64', or None.")


def _default_torch_device() -> torch.device:
    """Return the active default torch device, falling back to CPU when unavailable."""
    get_default_device = getattr(torch, "get_default_device", None)
    if get_default_device is None:
        return torch.device("cpu")
    return torch.device(get_default_device())


def _reduce_log_likelihood(
    ll: torch.Tensor,
    *,
    channel_agg: Literal["logmeanexp", "logsumexp", "first"],
    repetition_agg: Literal["logmeanexp", "logsumexp", "first"],
) -> torch.Tensor:
    """Reduce SPFlow log-likelihood tensor to per-sample scalar log-likelihoods.

    SPFlow modules typically return log-likelihoods shaped like:
        (batch, features, channels, repetitions)

    This function:
    - sums over features (log-space product),
    - aggregates repetitions and channels (mixture-like reduction), and
    - returns a 1D tensor of shape (batch,).
    """
    if ll.dim() == 2:
        ll = rearrange(ll, "b f -> b f 1 1")
    elif ll.dim() == 3:
        ll = rearrange(ll, "b f c -> b f c 1")
    elif ll.dim() != 4:
        raise InvalidParameterError(f"Unexpected log-likelihood shape {tuple(ll.shape)}.")

    if ll.shape[0] == 0:
        return ll.new_zeros((0,))

    ll = reduce(ll, "b f c r -> b c r", "sum")

    def reduce_over(t: torch.Tensor, dim: int, method: str) -> torch.Tensor:
        if t.shape[dim] == 1:
            return t.squeeze(dim)
        if method == "first":
            return t.select(dim, 0)
        if method == "logsumexp":
            return torch.logsumexp(t, dim=dim)
        if method == "logmeanexp":
            return torch.logsumexp(t, dim=dim) - math.log(t.shape[dim])
        raise InvalidParameterError(f"Unknown reduction method '{method}'.")

    ll = reduce_over(ll, dim=-1, method=repetition_agg)  # (B, C)
    ll = reduce_over(ll, dim=-1, method=channel_agg)  # (B,)
    return ll


@dataclass(frozen=True)
class _StructureLearnerSpec:
    name: Literal["learn_spn", "prometheus"]
    kwargs: dict[str, Any] | None


class SPFlowDensityEstimator(BaseEstimator, DensityMixin):
    """scikit-learn compatible density estimator for SPFlow models.

    Supports two workflows:
    - **Structure learning**: learn a model from data via `learn_spn` or `learn_prometheus`.
    - **Parameter fitting**: fit parameters of a provided SPFlow model via MLE.

    Args:
        model: Optional SPFlow model to fit and use for scoring/sampling.
        structure_learner: "learn_spn" or "prometheus". Only used when `model` is None.
        structure_learner_kwargs: Keyword arguments forwarded to the structure learner.
        fit_params: If True and `model` is provided, run MLE (`maximum_likelihood_estimation`) in `fit`.
        leaf: Leaf family used when learning structure and `model` is None. Currently supports "normal".
        leaf_out_channels: Output channels for the leaf module template (passed to `Normal`).
        min_instances_slice: Stopping criterion for structure learning (forwarded if not overridden).
        min_features_slice: Stopping criterion for structure learning (forwarded if not overridden).
        device: Torch device string (e.g., "cpu", "cuda"). If None, uses model device or the
            active PyTorch default device.
        dtype: Torch dtype string ("float32", "float64") for inputs.
        channel_agg: How to aggregate multiple output channels into a scalar log-likelihood.
        repetition_agg: How to aggregate multiple repetitions into a scalar log-likelihood.
    """

    def __init__(
        self,
        model: Module | None = None,
        *,
        structure_learner: Literal["learn_spn", "prometheus"] = "learn_spn",
        structure_learner_kwargs: dict[str, Any] | None = None,
        fit_params: bool = True,
        leaf: Literal["normal"] = "normal",
        leaf_out_channels: int = 1,
        min_instances_slice: int = 100,
        min_features_slice: int = 2,
        device: str | None = None,
        dtype: Literal["float32", "float64"] | None = None,
        channel_agg: Literal["logmeanexp", "logsumexp", "first"] = "logmeanexp",
        repetition_agg: Literal["logmeanexp", "logsumexp", "first"] = "logmeanexp",
    ) -> None:
        _require_sklearn()
        self.model = model
        self.structure_learner = structure_learner
        self.structure_learner_kwargs = structure_learner_kwargs
        self.fit_params = fit_params
        self.leaf = leaf
        self.leaf_out_channels = leaf_out_channels
        self.min_instances_slice = min_instances_slice
        self.min_features_slice = min_features_slice
        self.device = device
        self.dtype = dtype
        self.channel_agg = channel_agg
        self.repetition_agg = repetition_agg

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        if hasattr(self, "model_") and isinstance(self.model_, Module):
            return self.model_.device
        if self.model is not None:
            return self.model.device
        return _default_torch_device()

    def _to_tensor(self, x: Any) -> torch.Tensor:
        arr = _as_2d_numpy(x)
        dtype = _torch_dtype_from_str(self.dtype)
        return torch.as_tensor(arr, dtype=dtype, device=self._device())

    def _leaf_template(self, n_features: int) -> Any:
        if self.leaf != "normal":
            raise InvalidParameterError(f"Unknown leaf '{self.leaf}'.")
        return Normal(scope=Scope(list(range(n_features))), out_channels=self.leaf_out_channels).to(
            self._device()
        )

    def _structure_spec(self) -> _StructureLearnerSpec:
        if self.structure_learner not in ("learn_spn", "prometheus"):
            raise InvalidParameterError(
                "structure_learner must be 'learn_spn' or 'prometheus', " f"got '{self.structure_learner}'."
            )
        return _StructureLearnerSpec(name=self.structure_learner, kwargs=self.structure_learner_kwargs)

    def fit(self, X: Any, y: Any | None = None) -> "SPFlowDensityEstimator":
        """Fit a density model.

        Args:
            X: Array-like of shape (n_samples, n_features).
            y: Ignored. Present for scikit-learn compatibility.
        """
        del y
        x_tensor = self._to_tensor(X)
        self.n_features_in_ = int(x_tensor.shape[1])

        if self.model is None:
            leaf_modules = self._leaf_template(self.n_features_in_)
            spec = self._structure_spec()
            learner_kwargs: dict[str, Any] = {
                "out_channels": 1,
                "min_instances_slice": self.min_instances_slice,
                "min_features_slice": self.min_features_slice,
            }
            if spec.kwargs:
                learner_kwargs.update(spec.kwargs)

            if spec.name == "learn_spn":
                self.model_ = learn_spn(x_tensor, leaf_modules=leaf_modules, **learner_kwargs)
            else:
                self.model_ = learn_prometheus(x_tensor, leaf_modules=leaf_modules, **learner_kwargs)
        else:
            if not isinstance(self.model, Module):
                raise InvalidTypeError(
                    f"model must be a spflow.modules.module.Module, got {type(self.model)}."
                )
            self.model_ = self.model
            if self.fit_params:
                mle = getattr(self.model_, "maximum_likelihood_estimation", None)
                if mle is None:
                    raise InvalidParameterError(
                        "fit_params=True requires a model exposing maximum_likelihood_estimation "
                        "(typically leaf modules). "
                        "For general circuit models, use spflow.learn.expectation_maximization(...)."
                    )
                try:
                    mle(x_tensor)
                except UnsupportedOperationError as exc:
                    raise InvalidParameterError(
                        "fit_params=True requires a model exposing maximum_likelihood_estimation "
                        "(typically leaf modules). "
                        "For general circuit models, use spflow.learn.expectation_maximization(...)."
                    ) from exc

        return self

    def score_samples(self, X: Any) -> np.ndarray:
        """Compute per-sample log-likelihood under the fitted model."""
        check_is_fitted(self, attributes=["model_"])
        x_tensor = self._to_tensor(X)
        with torch.no_grad():
            ll = self.model_.log_likelihood(x_tensor)
            reduced = _reduce_log_likelihood(
                ll,
                channel_agg=self.channel_agg,
                repetition_agg=self.repetition_agg,
            )
        return reduced.detach().cpu().numpy()

    def sample(self, n_samples: int = 1, *, random_state: int | None = None) -> np.ndarray:
        """Generate samples from the fitted model.

        Args:
            n_samples: Number of samples to draw.
            random_state: Optional seed for deterministic sampling.
        """
        check_is_fitted(self, attributes=["model_"])
        if not isinstance(n_samples, int) or n_samples < 1:
            raise InvalidParameterError(f"n_samples must be a positive integer, got {n_samples}.")

        if random_state is not None and not isinstance(random_state, (int, np.integer)):
            raise InvalidTypeError(f"random_state must be an int or None, got {type(random_state)}.")

        seed = int(random_state) if random_state is not None else None
        device = self._device()
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [device.index or 0]

        with torch.random.fork_rng(devices=cuda_devices):
            if seed is not None:
                torch.manual_seed(seed)
            with torch.no_grad():
                samples = self.model_.sample(num_samples=n_samples)
        return samples.detach().cpu().numpy()


class SPFlowClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn compatible classifier wrapper for SPFlow classifiers.

    This wrapper delegates to a provided SPFlow model that implements
    `predict_proba(torch.Tensor) -> torch.Tensor`.
    """

    def __init__(
        self,
        model: Any,
        *,
        device: str | None = None,
        dtype: Literal["float32", "float64"] | None = None,
    ) -> None:
        _require_sklearn()
        self.model = model
        self.device = device
        self.dtype = dtype

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        if hasattr(self.model, "device"):
            return torch.device(getattr(self.model, "device"))
        return _default_torch_device()

    def _to_tensor(self, x: Any) -> torch.Tensor:
        arr = _as_2d_numpy(x)
        dtype = _torch_dtype_from_str(self.dtype)
        return torch.as_tensor(arr, dtype=dtype, device=self._device())

    def fit(self, X: Any, y: Any) -> "SPFlowClassifier":
        """Store class labels for sklearn compatibility."""
        del X
        classes = np.unique(np.asarray(y))
        self.classes_ = classes
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["classes_"])
        x_tensor = self._to_tensor(X)
        with torch.no_grad():
            probs = self.model.predict_proba(x_tensor)
        return probs.detach().cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels using argmax over predicted probabilities."""
        check_is_fitted(self, attributes=["classes_"])
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return np.asarray(self.classes_)[indices]
