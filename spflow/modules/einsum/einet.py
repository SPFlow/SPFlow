"""Einet (EinsumNetworks) module for efficient deep probabilistic models.

Einet provides a scalable architecture for Sum-Product Networks using
EinsumLayer or LinsumLayer for efficient batched computations.

Reference:
    Peharz, R., et al. (2020). "Einsum Networks: Fast and Scalable Learning of
    Tractable Probabilistic Circuits." ICML 2020.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
from torch import nn

from spflow.interfaces.classifier import Classifier
from spflow.meta.data.scope import Scope
from spflow.modules.einsum.einsum_layer import EinsumLayer
from spflow.modules.einsum.linsum_layer import LinsumLayer
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.rat.factorize import Factorize
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache, cached
from spflow.utils.inference import log_posterior
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class Einet(Module, Classifier):
    """Einsum Network (Einet) for scalable deep probabilistic modeling.

    Einet uses efficient einsum-based layers (EinsumLayer or LinsumLayer) to
    combine product and sum operations, enabling faster training and inference
    compared to traditional RAT-SPNs.

    Attributes:
        leaf_modules (list[LeafModule]): Leaf distribution modules.
        num_classes (int): Number of output classes (root sum nodes).
        num_sums (int): Number of sum nodes per intermediate layer.
        num_leaves (int): Number of leaf distribution components.
        depth (int): Number of einsum layers.
        num_repetitions (int): Number of parallel circuit repetitions.
        layer_type (str): Type of intermediate layer ("einsum" or "linsum").
        structure (str): Structure building mode ("top-down" or "bottom-up").

    Reference:
        Peharz, R., et al. (2020). "Einsum Networks: Fast and Scalable Learning
        of Tractable Probabilistic Circuits." ICML 2020.
    """

    def __init__(
        self,
        leaf_modules: list[LeafModule],
        num_classes: int = 1,
        num_sums: int = 10,
        num_leaves: int = 10,
        depth: int = 1,
        num_repetitions: int = 5,
        layer_type: Literal["einsum", "linsum"] = "linsum",
        structure: Literal["top-down", "bottom-up"] = "top-down",
    ) -> None:
        """Initialize Einet with specified architecture parameters.

        Args:
            leaf_modules: Leaf distribution modules forming the base layer.
            num_classes: Number of root sum nodes (classes). Defaults to 1.
            num_sums: Number of sum nodes per intermediate layer. Defaults to 10.
            num_leaves: Number of leaf distribution components. Defaults to 10.
            depth: Number of einsum layers. Defaults to 1.
            num_repetitions: Number of parallel circuit repetitions. Defaults to 5.
            layer_type: Type of intermediate layer ("einsum" or "linsum").
                Defaults to "linsum".
            structure: Structure building mode ("top-down" or "bottom-up").
                Defaults to "top-down".

        Raises:
            ValueError: If architectural parameters are invalid.
        """
        super().__init__()

        # Validate parameters
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if num_sums < 1:
            raise ValueError(f"num_sums must be >= 1, got {num_sums}")
        if num_leaves < 1:
            raise ValueError(f"num_leaves must be >= 1, got {num_leaves}")
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        if num_repetitions < 1:
            raise ValueError(f"num_repetitions must be >= 1, got {num_repetitions}")
        if layer_type not in ("einsum", "linsum"):
            raise ValueError(f"layer_type must be 'einsum' or 'linsum', got {layer_type}")
        if structure not in ("top-down", "bottom-up"):
            raise ValueError(f"structure must be 'top-down' or 'bottom-up', got {structure}")

        # Store configuration
        self.leaf_modules = nn.ModuleList(leaf_modules)
        self.num_classes = num_classes
        self.num_sums = num_sums
        self.num_leaves = num_leaves
        self.depth = depth
        self.num_repetitions = num_repetitions
        self.layer_type = layer_type
        self.structure = structure

        # Compute scope from leaf modules
        self.scope = Scope.join_all([leaf.scope for leaf in leaf_modules])
        self.num_features = len(self.scope.query)

        # Validate depth against number of features
        if 2**depth > self.num_features:
            raise ValueError(
                f"depth {depth} too large for {self.num_features} features. "
                f"Maximum depth is {int(np.floor(np.log2(self.num_features)))}."
            )

        # Build the architecture
        if structure == "top-down":
            self._build_structure_top_down()
        else:
            self._build_structure_bottom_up()

        # Shape computation
        self.in_shape = self.root_node.in_shape
        self.out_shape = self.root_node.out_shape

    def _get_layer_class(self):
        """Get the layer class based on layer_type."""
        if self.layer_type == "einsum":
            return EinsumLayer
        else:
            return LinsumLayer

    def _build_structure_top_down(self) -> None:
        """Build Einet structure from top (root) to bottom (leaves).

        In top-down mode, we define layers starting from the root and work
        down to the leaves. Each layer i has 2^i input features.
        """
        LayerClass = self._get_layer_class()
        layers: list[Module] = []

        # Build layers from top (i=1) to bottom (i=depth)
        for i in range(1, self.depth + 1):
            # Number of input channels
            if i < self.depth:
                in_channels = self.num_sums
            else:
                in_channels = self.num_leaves

            # Number of output channels
            if i == 1:
                out_channels = self.num_classes
            else:
                out_channels = self.num_sums

            # Number of features at this layer
            in_features = 2**i

            # Create placeholder input with correct shape for layer construction
            # We'll connect them properly after building all layers
            layers.append({
                "in_features": in_features,
                "in_channels": in_channels,
                "out_channels": out_channels,
            })

        # Handle depth=0 case: single sum layer
        if self.depth == 0:
            # Create factorized leaves with single output feature
            fac_layer = Factorize(
                inputs=list(self.leaf_modules),
                depth=0,
                num_repetitions=self.num_repetitions,
            )
            # Single sum layer from leaves to root
            root = Sum(
                inputs=fac_layer,
                out_channels=self.num_classes,
                num_repetitions=self.num_repetitions,
            )
        else:
            # Create factorized leaves
            leaf_num_features_out = 2**self.depth
            fac_layer = Factorize(
                inputs=list(self.leaf_modules),
                depth=self.depth,
                num_repetitions=self.num_repetitions,
            )

            # Build layers bottom-up (reverse of how they process data)
            current = fac_layer
            for i in range(self.depth, 0, -1):
                layer_info = layers[i - 1]

                # Create einsum/linsum layer
                current = LayerClass(
                    inputs=current,
                    out_channels=layer_info["out_channels"],
                    num_repetitions=self.num_repetitions,
                )

            root = current

        # Mix repetitions if we have multiple
        if self.num_repetitions > 1:
            root = RepetitionMixingLayer(
                inputs=root,
                out_channels=self.num_classes,
                num_repetitions=self.num_repetitions,
            )

        # Final root sum if multiple classes
        if self.num_classes > 1 and not isinstance(root, RepetitionMixingLayer):
            self.root_node = Sum(inputs=root, out_channels=1, num_repetitions=1)
        else:
            self.root_node = root

        # Store layers for access
        self.factorize = fac_layer

    def _build_structure_bottom_up(self) -> None:
        """Build Einet structure from bottom (leaves) to top (root).

        In bottom-up mode, we start with the full feature set and
        progressively halve features at each layer.
        """
        LayerClass = self._get_layer_class()

        # Create factorized leaves (no depth reduction, just random permutations)
        fac_layer = Factorize(
            inputs=list(self.leaf_modules),
            depth=int(np.log2(self.num_features)) if self.num_features > 1 else 0,
            num_repetitions=self.num_repetitions,
        )

        # Generate random permutations for each repetition
        permutations = torch.empty(
            (self.num_repetitions, self.num_features), dtype=torch.long
        )
        for r in range(self.num_repetitions):
            permutations[r] = torch.randperm(self.num_features)
        self.register_buffer("permutation", permutations)

        # Build layers from leaves to root
        current = fac_layer
        in_features = fac_layer.out_shape.features

        for i in range(self.depth):
            # Input channels
            if i == 0:
                in_channels = self.num_leaves
            else:
                in_channels = self.num_sums

            # Output channels
            out_channels = self.num_sums

            # Create einsum/linsum layer
            current = LayerClass(
                inputs=current,
                out_channels=out_channels,
                num_repetitions=self.num_repetitions,
            )

            in_features = current.out_shape.features

        # Handle depth=0 case
        if self.depth == 0:
            # Single sum layer
            current = Sum(
                inputs=fac_layer,
                out_channels=self.num_sums,
                num_repetitions=self.num_repetitions,
            )

        # Final root sum to reduce features to 1
        if current.out_shape.features > 1:
            # Need a sum layer to combine all features
            root = Sum(
                inputs=current,
                out_channels=self.num_classes,
                num_repetitions=self.num_repetitions,
            )
        else:
            root = current

        # Mix repetitions if we have multiple
        if self.num_repetitions > 1:
            root = RepetitionMixingLayer(
                inputs=root,
                out_channels=self.num_classes,
                num_repetitions=self.num_repetitions,
            )

        # Final root sum if multiple classes
        if self.num_classes > 1 and not isinstance(root, RepetitionMixingLayer):
            self.root_node = Sum(inputs=root, out_channels=1, num_repetitions=1)
        else:
            self.root_node = root

        # Store layers for access
        self.factorize = fac_layer

    @property
    def n_out(self) -> int:
        """Number of output nodes."""
        return 1

    @property
    def feature_to_scope(self) -> np.ndarray:
        """Mapping from output features to their scopes."""
        return self.root_node.feature_to_scope

    @property
    def scopes_out(self) -> list[Scope]:
        """Output scopes."""
        return self.root_node.scopes_out

    @cached
    def log_likelihood(
        self,
        data: torch.Tensor,
        cache: Cache | None = None,
    ) -> torch.Tensor:
        """Compute log-likelihood for input data.

        Args:
            data: Input data tensor of shape (batch_size, num_features).
            cache: Optional cache for intermediate results.

        Returns:
            Log-likelihood tensor of shape (batch_size, 1, num_classes, 1).
        """
        # Apply permutation if in bottom-up mode
        if hasattr(self, "permutation") and self.structure == "bottom-up":
            # Permute features for each repetition
            # This is handled inside the factorize layer via its indices
            pass

        return self.root_node.log_likelihood(data, cache=cache)

    def log_posterior(
        self,
        data: torch.Tensor,
        cache: Cache | None = None,
    ) -> torch.Tensor:
        """Compute log-posterior probabilities for multi-class models.

        Args:
            data: Input data tensor.
            cache: Optional cache for intermediate results.

        Returns:
            Log-posterior probabilities of shape (batch_size, num_classes).

        Raises:
            ValueError: If model has only one class.
        """
        if self.num_classes <= 1:
            raise ValueError("Posterior can only be computed for models with multiple classes.")

        ll_y = self.root_node.log_weights
        ll_y = ll_y.squeeze().view(1, -1)
        ll = self.root_node.inputs.log_likelihood(data, cache=cache)
        ll = ll.squeeze(-1).squeeze(1)
        return log_posterior(log_likelihood=ll, log_prior=ll_y)

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.

        Args:
            data: Input data tensor.

        Returns:
            Class probabilities of shape (batch_size, num_classes).
        """
        log_post = self.log_posterior(data)
        return torch.exp(log_post)

    def sample(
        self,
        num_samples: int | None = None,
        data: torch.Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> torch.Tensor:
        """Generate samples from the Einet.

        Args:
            num_samples: Number of samples to generate.
            data: Optional data tensor with NaN values to impute.
            is_mpe: Whether to perform MPE (most probable explanation).
            cache: Optional cache for intermediate results.
            sampling_ctx: Optional sampling context.

        Returns:
            Sampled tensor.
        """
        # Handle num_samples case
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full(
                (num_samples, self.num_features), torch.nan, device=self.device
            )

        # Initialize sampling context
        if sampling_ctx is None and self.num_classes > 1:
            sampling_ctx = init_default_sampling_context(
                sampling_ctx, data.shape[0], data.device
            )
            logits = self.root_node.logits
            if logits.shape != (1, self.num_classes, 1):
                raise ValueError(
                    f"Expected logits shape (1, {self.num_classes}, 1), got {logits.shape}"
                )
            logits = logits.squeeze(-1)
            logits = logits.unsqueeze(0).expand(data.shape[0], -1, -1)

            if is_mpe:
                sampling_ctx.channel_index = torch.argmax(logits, dim=-1)
            else:
                sampling_ctx.channel_index = torch.distributions.Categorical(
                    logits=logits
                ).sample()
        else:
            sampling_ctx = init_default_sampling_context(
                sampling_ctx, data.shape[0], data.device
            )

        # Sample from appropriate root
        if self.num_classes > 1:
            sample_root = self.root_node.inputs
        else:
            sample_root = self.root_node

        return sample_root.sample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

    def expectation_maximization(
        self,
        data: torch.Tensor,
        cache: Cache | None = None,
    ) -> None:
        """Perform expectation-maximization step.

        Args:
            data: Input data tensor.
            cache: Optional cache with log-likelihoods.
        """
        self.root_node.expectation_maximization(data, cache=cache)

    def maximum_likelihood_estimation(
        self,
        data: torch.Tensor,
        weights: torch.Tensor | None = None,
        cache: Cache | None = None,
    ) -> None:
        """Update parameters via maximum likelihood estimation.

        Args:
            data: Input data tensor.
            weights: Optional sample weights.
            cache: Optional cache.
        """
        self.root_node.maximum_likelihood_estimation(data, weights=weights, cache=cache)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: Random variable indices to marginalize.
            prune: Whether to prune redundant modules.
            cache: Optional cache.

        Returns:
            Marginalized module or None if fully marginalized.
        """
        return self.root_node.marginalize(marg_rvs, prune=prune, cache=cache)

    def extra_repr(self) -> str:
        """String representation of module configuration."""
        return (
            f"num_features={self.num_features}, num_classes={self.num_classes}, "
            f"num_sums={self.num_sums}, num_leaves={self.num_leaves}, "
            f"depth={self.depth}, num_repetitions={self.num_repetitions}, "
            f"layer_type={self.layer_type}, structure={self.structure}"
        )
