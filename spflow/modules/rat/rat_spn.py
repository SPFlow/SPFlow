"""Random and Tensorized Sum-Product Networks (RAT-SPNs) implementation.

RAT-SPNs provide a principled approach to building deep probabilistic models
through randomized circuit construction, combining interpretability with
expressiveness through tensorized operations.

Reference:
    Peharz, R., et al. (2020). "Random Sum-Product Networks: A Simple and
    Effective Approach to Probabilistic Deep Learning." NeurIPS 2020.
"""

from __future__ import annotations

import numpy as np
import torch

from spflow.interfaces.classifier import Classifier
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.ops.split import Split, SplitMode
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.modules.ops.split_interleaved import SplitInterleaved
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.modules.products.outer_product import OuterProduct
from spflow.modules.rat.factorize import Factorize
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache, cached
from spflow.utils.inference import log_posterior
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class RatSPN(Module, Classifier):
    """Random and Tensorized Sum-Product Network (RAT-SPN).

    Scalable deep probabilistic model with randomized circuit construction.
    Consists of alternating sum (region) and product (partition) layers that
    recursively partition input space. Random construction prevents overfitting
    while maintaining tractable exact inference.

    Attributes:
        leaf_modules (list[LeafModule]): Leaf distribution modules.
        n_root_nodes (int): Number of root sum nodes.
        n_region_nodes (int): Number of sum nodes per region.
        depth (int): Number of partition/region layers.
        num_repetitions (int): Number of parallel circuit instances.
        scope (Scope): Combined scope of all leaf modules.

    Reference:
        Peharz, R., et al. (2020). "Random Sum-Product Networks: A Simple and
        Effective Approach to Probabilistic Deep Learning." NeurIPS 2020.
    """

    def __init__(
        self,
        leaf_modules: list[LeafModule],
        n_root_nodes: int,
        n_region_nodes: int,
        num_repetitions: int,
        depth: int,
        outer_product: bool | None = False,
        split_mode: SplitMode | None = None,
        num_splits: int | None = 2,
    ) -> None:
        """Initialize RAT-SPN with specified architecture parameters.

        Creates a Random and Tensorized SPN by recursively constructing layers of
        sum and product nodes. Circuit structure is fixed after initialization.

        Args:
            leaf_modules (list[LeafModule]): Leaf distributions forming the base layer.
            n_root_nodes (int): Number of root sum nodes in final mixture.
            n_region_nodes (int): Number of sum nodes in each region layer.
            num_repetitions (int): Number of parallel circuit instances.
            depth (int): Number of partition/region layers.
            outer_product (bool | None, optional): Use outer product instead of
                elementwise product for partitions. Defaults to False.
            split_mode (SplitMode | None, optional): Split configuration.
                Use SplitMode.consecutive() or SplitMode.interleaved().
                Defaults to SplitMode.consecutive(num_splits) if not specified.
            num_splits (int | None, optional): Number of splits in each partition.
                Must be at least 2. Defaults to 2.

        Raises:
            ValueError: If architectural parameters are invalid.
        """
        super().__init__()
        self.n_root_nodes = n_root_nodes
        self.n_region_nodes = n_region_nodes
        self.n_leaf_nodes = leaf_modules[0].out_shape.channels
        self.leaf_modules = leaf_modules
        self.depth = depth
        self.num_repetitions = num_repetitions
        self.outer_product = outer_product
        self.num_splits = num_splits
        self.split_mode = split_mode if split_mode is not None else SplitMode.consecutive(num_splits)
        self.scope = Scope.join_all([leaf.scope for leaf in leaf_modules])

        if n_root_nodes < 1:
            raise ValueError(f"Specified value of 'n_root_nodes' must be at least 1, but is {n_root_nodes}.")
        if n_region_nodes < 1:
            raise ValueError(
                f"Specified value for 'n_region_nodes' must be at least 1, but is {n_region_nodes}."
            )
        if self.n_leaf_nodes < 1:
            raise ValueError(
                f"Specified value for 'n_leaf_nodes' must be at least 1, but is {self.n_leaf_nodes}."
            )

        if self.num_splits < 2:
            raise ValueError(
                f"Specified value for 'num_splits' must be at least 2, but is {self.num_splits}."
            )

        self.create_spn()
        
        # Shape computation: delegate to root node
        self.in_shape = self.root_node.in_shape
        self.out_shape = self.root_node.out_shape


    def create_spn(self):
        """Create the RAT-SPN architecture.

        Builds the RAT-SPN circuit structure from bottom to top based on
        the provided architectural parameters. Architecture is constructed recursively from
        leaves to root using alternating layers of sum and product nodes, and the final
        structure depends on depth and branching parameters.
        """
        if self.outer_product:
            product_layer = OuterProduct
        else:
            product_layer = ElementwiseProduct
        # Factorize the leaves modules
        fac_layer = Factorize(
            inputs=self.leaf_modules, depth=self.depth, num_repetitions=self.num_repetitions
        )
        depth = self.depth
        root = None

        for i in range(depth):
            # Create the lowest layer with the factorized leaves modules as input
            if i == 0:
                out_prod = product_layer(inputs=self.split_mode.create(fac_layer))
                if depth == 1:
                    sum_layer = Sum(
                        inputs=out_prod, out_channels=self.n_root_nodes, num_repetitions=self.num_repetitions
                    )
                else:
                    sum_layer = Sum(
                        inputs=out_prod,
                        out_channels=self.n_region_nodes,
                        num_repetitions=self.num_repetitions,
                    )
                root = sum_layer

            # Special case for the last intermediate layer
            elif i == depth - 1:
                out_prod = product_layer(self.split_mode.create(root))
                sum_layer = Sum(
                    inputs=out_prod, out_channels=self.n_root_nodes, num_repetitions=self.num_repetitions
                )
                root = sum_layer
            # Create the intermediate layers
            else:
                out_prod = product_layer(self.split_mode.create(root))
                sum_layer = Sum(
                    inputs=out_prod, out_channels=self.n_region_nodes, num_repetitions=self.num_repetitions
                )
                root = sum_layer

        # MixingLayer: Sums over repetitions
        root = RepetitionMixingLayer(
            inputs=root, out_channels=self.n_root_nodes, num_repetitions=self.num_repetitions
        )

        # root node: Sum over all out_channels
        if self.n_root_nodes > 1:
            self.root_node = Sum(inputs=root, out_channels=1, num_repetitions=1)
        else:
            self.root_node = root

    @property
    def n_out(self) -> int:
        return 1

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.root_node.feature_to_scope

    @property
    def scopes_out(self) -> list[Scope]:
        return self.root_node.scopes_out

    @cached
    def log_likelihood(
        self,
        data: torch.Tensor,
        cache: Cache | None = None,
    ) -> torch.Tensor:
        """Compute log likelihood for RAT-SPN.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary for caching intermediate results.

        Returns:
            Log-likelihood values.
        """
        ll = self.root_node.log_likelihood(
            data,
            cache=cache,
        )
        return ll

    def log_posterior(
        self,
        data: torch.Tensor,
        cache: Cache | None = None,
    ) -> torch.Tensor:
        """Compute log-posterior probabilities for multi-class models.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary for caching intermediate results.

        Returns:
            Log-posterior probabilities.

        Raises:
            ValueError: If model has only one root node (single class).
        """
        if self.n_root_nodes <= 1:
            raise ValueError("Posterior can only be computed for models with multiple classes.")

        ll_y = self.root_node.log_weights  # shape: (1, n_root_nodes, 1, 1)
        ll_y = ll_y.squeeze().view(1, -1)  # shape: (1, n_root_nodes)
        ll = self.root_node.inputs.log_likelihood(
            data,
            cache=cache,
        )  # shape: (batch_size,1 , n_root_nodes)
        ll = ll.squeeze(-1).squeeze(1)  # remove repetition and feature dimensions -> shape: (batch_size, n_root_nodes)
        return log_posterior(log_likelihood=ll, log_prior=ll_y)

    def predict_proba(self, data: torch.Tensor):
        """Classify input data using RAT-SPN.

        Args:
            data: Input data tensor.
        Returns:
            Predicted class labels.
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
        """Generate samples from the RAT-SPN.

        Args:
            num_samples: Number of samples to generate.
            data: Data tensor with NaN values to fill with samples.
            is_mpe: Whether to perform maximum a posteriori estimation.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Sampled values.
        """

        # Handle num_samples case (create empty data tensor)
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), torch.nan, device=self.device)

        # if no sampling context is provided, initialize a context by sampling from the root node
        if sampling_ctx is None and self.n_root_nodes > 1:
            sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)
            logits = self.root_node.logits
            if logits.shape != (1, self.n_root_nodes, 1):
                raise ValueError(f"Expected logits shape (1, {self.n_root_nodes}, 1), but got {logits.shape}")
            logits = logits.squeeze(-1)
            logits = logits.unsqueeze(0).expand(data.shape[0], -1, -1)  # shape [b ,1, n_root_nodes]

            if is_mpe:
                sampling_ctx.channel_index = torch.argmax(logits, dim=-1)
            else:
                sampling_ctx.channel_index = torch.distributions.Categorical(logits=logits).sample()

        else:
            sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        # if the model only has one root node, we can directly sample from the mixing layer
        if self.n_root_nodes > 1:
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
            cache: Optional cache dictionary.
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
            cache: Optional cache dictionary.
        """
        self.root_node.maximum_likelihood_estimation(
            data,
            weights=weights,
            cache=cache,
        )

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variables to marginalize.
            prune: Whether to prune the module.
            cache: Optional cache dictionary.

        Returns:
            Marginalized module or None.
        """
        return self.root_node.marginalize(marg_rvs, prune=prune, cache=cache)
