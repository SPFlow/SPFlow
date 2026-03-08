"""Naive Bayes circuits built from SPFlow leaf modules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.interfaces.classifier import Classifier
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache, cached
from spflow.utils.inference import log_posterior
from spflow.utils.sampling_context import SamplingContext


class NaiveBayes(Module, Classifier):
    """Naive Bayes model built from SPFlow leaves and a product root.

    Density estimation is represented by ``leaf layer -> Product``.
    Classification is represented by ``leaf layer -> Product -> Sum`` where the
    product captures ``p(x | y)`` across output channels and the root sum stores
    the class prior.
    """

    def __init__(
        self,
        leaf_modules: LeafModule | list[LeafModule],
        *,
        num_classes: int = 1,
        class_prior: Tensor | list[float] | None = None,
        learnable_prior: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise InvalidParameterError(f"num_classes must be >= 1, got {num_classes}.")

        leaves = self._normalize_leaf_modules(leaf_modules)
        self.leaf_modules = nn.ModuleList(leaves)
        self.num_classes = int(num_classes)
        self.learnable_prior = bool(learnable_prior)

        self._validate_leaves(leaves=leaves)
        self.scope = Scope.join_all([leaf.scope for leaf in leaves])

        if self.num_classes == 1:
            if class_prior is not None:
                raise InvalidParameterError("class_prior is only valid when num_classes > 1.")
            if self.learnable_prior:
                raise InvalidParameterError("learnable_prior requires num_classes > 1.")

        self.conditional_root = Product(inputs=list(self.leaf_modules))
        if self.num_classes > 1:
            prior_weights = self._build_prior_weights(class_prior=class_prior)
            self.root_node = Sum(
                inputs=self.conditional_root,
                out_channels=1,
                num_repetitions=1,
                weights=prior_weights,
            )
            self.root_node.logits.requires_grad_(self.learnable_prior)
        else:
            self.root_node = self.conditional_root

        self.in_shape = self.root_node.in_shape
        self.out_shape = self.root_node.out_shape

    @staticmethod
    def _normalize_leaf_modules(leaf_modules: LeafModule | list[LeafModule]) -> list[LeafModule]:
        if isinstance(leaf_modules, LeafModule):
            return [leaf_modules]
        if not isinstance(leaf_modules, list) or len(leaf_modules) == 0:
            raise InvalidParameterError(
                "leaf_modules must be a LeafModule or a non-empty list of LeafModules."
            )
        return leaf_modules

    def _validate_leaves(self, *, leaves: list[LeafModule]) -> None:
        expected_channels = self.num_classes if self.num_classes > 1 else 1
        for leaf in leaves:
            if not isinstance(leaf, LeafModule):
                raise InvalidParameterError(
                    f"NaiveBayes expects LeafModule inputs, got {type(leaf).__name__}."
                )
            if leaf.out_shape.repetitions != 1:
                raise InvalidParameterError(
                    "NaiveBayes currently supports only num_repetitions=1 for all leaves."
                )
            if leaf.out_shape.channels != expected_channels:
                raise InvalidParameterError(
                    f"Expected leaf out_channels={expected_channels}, got {leaf.out_shape.channels}."
                )

    def _build_prior_weights(self, class_prior: Tensor | list[float] | None) -> Tensor:
        if class_prior is None:
            prior = torch.full((self.num_classes,), 1.0 / self.num_classes, dtype=torch.get_default_dtype())
        else:
            prior = torch.as_tensor(class_prior, dtype=torch.get_default_dtype())
            if prior.ndim != 1 or prior.shape[0] != self.num_classes:
                raise InvalidParameterError(
                    f"class_prior must have shape ({self.num_classes},), got {tuple(prior.shape)}."
                )
            if not torch.isfinite(prior).all():
                raise InvalidParameterError("class_prior must contain only finite values.")
            if torch.any(prior <= 0):
                raise InvalidParameterError("class_prior must contain only strictly positive values.")
            prior = prior / prior.sum()

        return rearrange(prior, "co -> 1 co 1 1")

    @staticmethod
    def _normalize_empirical_prior(counts: Tensor) -> Tensor:
        eps = torch.finfo(counts.dtype).eps
        counts = counts + eps
        return counts / counts.sum()

    def _validate_targets(self, targets: Tensor, *, batch_size: int) -> Tensor:
        if not isinstance(targets, Tensor):
            raise InvalidParameterError("targets must be a torch.Tensor of class indices.")
        if targets.ndim != 1 or targets.shape[0] != batch_size:
            raise InvalidParameterError(
                f"targets must have shape ({batch_size},), got {tuple(targets.shape)}."
            )
        if torch.is_floating_point(targets) or torch.is_complex(targets) or targets.dtype == torch.bool:
            raise InvalidParameterError("targets must contain integer class indices.")

        target_indices = targets.to(device=self.root_node.device, dtype=torch.long)
        if torch.any(target_indices < 0) or torch.any(target_indices >= self.num_classes):
            raise InvalidParameterError(f"targets must contain class indices in [0, {self.num_classes - 1}].")
        return target_indices

    @property
    def n_out(self) -> int:
        return 1

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.root_node.feature_to_scope

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        return self.root_node.log_likelihood(data, cache=cache)

    def log_posterior(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        if self.num_classes <= 1:
            raise UnsupportedOperationError(
                "Posterior can only be computed for models with multiple classes."
            )

        ll_y = rearrange(self.root_node.log_weights, "1 co 1 1 -> 1 co")
        ll = rearrange(self.conditional_root.log_likelihood(data, cache=cache), "b 1 co 1 -> b co")
        return log_posterior(log_likelihood=ll, log_prior=ll_y)

    def predict_proba(self, data: Tensor) -> Tensor:
        return self.log_posterior(data).exp()

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        *,
        targets: Tensor | None = None,
        bias_correction: bool = True,
        nan_strategy: str | Callable | None = None,
    ) -> None:
        if self.num_classes == 1:
            if targets is not None:
                raise InvalidParameterError("targets must be None when num_classes == 1.")
            for leaf in self.leaf_modules:
                leaf.maximum_likelihood_estimation(
                    data,
                    bias_correction=bias_correction,
                    nan_strategy=nan_strategy,
                )
            return

        if targets is None:
            raise InvalidParameterError("targets are required for classifier maximum_likelihood_estimation.")

        target_indices = self._validate_targets(targets, batch_size=int(data.shape[0]))
        class_weights = F.one_hot(target_indices, num_classes=self.num_classes).to(
            device=data.device,
            dtype=data.dtype,
        )
        class_weights = rearrange(class_weights, "b co -> b 1 co 1")

        for leaf in self.leaf_modules:
            expanded_weights = class_weights.expand(-1, leaf.out_shape.features, -1, -1)
            leaf.maximum_likelihood_estimation(
                data,
                weights=expanded_weights,
                bias_correction=bias_correction,
                nan_strategy=nan_strategy,
            )

        if self.learnable_prior:
            counts = torch.bincount(target_indices, minlength=self.num_classes).to(
                device=self.root_node.device,
                dtype=self.root_node.logits.dtype,
            )
            prior = self._normalize_empirical_prior(counts)
            self.root_node.weights = rearrange(prior, "co -> 1 co 1 1")

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        return self.root_node._sample(data=data, sampling_ctx=sampling_ctx, cache=cache)

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        if self.num_classes == 1 or self.learnable_prior:
            self.root_node._expectation_maximization_step(
                data=data,
                bias_correction=bias_correction,
                cache=cache,
            )
            return

        self.conditional_root._expectation_maximization_step(
            data=data,
            bias_correction=bias_correction,
            cache=cache,
        )

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        return self.root_node.marginalize(marg_rvs, prune=prune, cache=cache)
