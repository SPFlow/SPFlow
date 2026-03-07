"""CNet (Cutset Network) structure learning algorithm.

Cutset Networks are probabilistic models that use recursive conditioning
on discrete variables to build OR-tree structures with Chow-Liu tree leaves.
"""

from __future__ import annotations

from typing import Sequence

import torch
from einops import repeat
from torch import Tensor

from spflow.exceptions import InvalidParameterError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum


def _default_generator() -> torch.Generator:
    """Create a generator on the active default device."""
    get_default_device = getattr(torch, "get_default_device", None)
    device = get_default_device() if get_default_device is not None else "cpu"
    return torch.Generator(device=device)


def _validate_discrete_data(data: Tensor, cardinalities: list[int], scope: Scope) -> None:
    """Validate that data contains valid discrete values."""
    if data.dim() != 2:
        raise InvalidParameterError(f"Data must be 2D (N, D), got shape {tuple(data.shape)}.")

    if torch.isnan(data).any():
        raise InvalidParameterError("learn_cnet requires complete data without NaNs.")

    for var in scope.query:
        col = data[:, var]
        if not torch.allclose(col, col.round()):
            raise InvalidParameterError(f"Variable {var} contains non-integer values.")
        if col.min() < 0:
            raise InvalidParameterError(f"Variable {var} contains negative values.")
        if col.max() >= cardinalities[var]:
            raise InvalidParameterError(f"Variable {var} has values >= cardinality {cardinalities[var]}.")


def _compute_entropy(data: Tensor, var: int, K: int) -> float:
    """Compute entropy of a variable's distribution in data."""
    if data.shape[0] == 0:
        return 0.0
    col = data[:, var].long()
    counts = torch.bincount(col, minlength=K).float()
    probs = counts / counts.sum().clamp_min(1e-12)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum()
    return float(entropy.item())


def _select_conditioning_variable_naive_mle(data: Tensor, scope: Scope, cardinalities: list[int]) -> int:
    """Select variable with highest entropy (most balanced splits)."""
    best_var = scope.query[0]
    best_entropy = -float("inf")

    for var in scope.query:
        K = cardinalities[var]
        entropy = _compute_entropy(data, var, K)
        if entropy > best_entropy:
            best_entropy = entropy
            best_var = var

    return best_var


def _select_conditioning_variable_random(scope: Scope, rng: torch.Generator) -> int:
    """Select a random variable from scope."""
    idx = int(torch.randint(len(scope.query), (1,), generator=rng).item())
    return scope.query[idx]


def learn_cnet(
    data: Tensor,
    *,
    cardinalities: int | Sequence[int],
    scope: Scope | None = None,
    cond: str = "naive_mle",
    min_instances_slice: int = 20,
    min_features_slice: int = 1,
    out_channels: int = 1,
    alpha: float = 0.01,
    seed: int | None = None,
) -> Module:
    """Learn a Cutset Network (CNet) from discrete data.

    CNets recursively condition on discrete variables to build an OR-tree
    structure with Chow-Liu tree (CLTree) leaves at terminals.

    Args:
        data: Tensor of shape (N, D) with discrete integer values.
        cardinalities: Number of categories per variable. Either a single int
            (same cardinality for all variables) or a sequence of length D.
        scope: Variable scope for the CNet. If None, inferred from data.
        cond: Conditioning strategy. One of:
            - "naive_mle": Greedy selection by highest entropy (most balanced splits)
            - "random": Random selection (seeded by `seed`)
        min_instances_slice: Minimum instances before creating a leaf (default: 20).
        min_features_slice: Minimum features before creating a leaf (default: 1).
        out_channels: Number of output channels (default: 1).
        alpha: Smoothing parameter for CLTree leaves (default: 0.01).
        seed: Random seed for reproducibility (used with cond="random").

    Returns:
        A Module representing the learned CNet structure.

    Raises:
        InvalidParameterError: If data or parameters are invalid.
        ValueError: If cond strategy is unknown.
    """
    if data.dim() != 2:
        raise InvalidParameterError(f"Data must be 2D (N, D), got shape {tuple(data.shape)}.")

    N, D = data.shape

    # Normalize cardinalities to list
    if isinstance(cardinalities, int):
        card_list = [cardinalities] * D
    else:
        card_list = list(cardinalities)
        if len(card_list) != D:
            raise InvalidParameterError(f"cardinalities length {len(card_list)} != data columns {D}.")

    # Infer scope if not provided
    if scope is None:
        scope = Scope(list(range(D)))

    # Validate conditioning strategy
    if cond not in ("naive_mle", "random"):
        raise ValueError(f"Unknown conditioning strategy: {cond}. Use 'naive_mle' or 'random'.")

    # Validate data
    _validate_discrete_data(data, card_list, scope)

    # Set up RNG for random conditioning
    rng = _default_generator()
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.manual_seed(int(torch.randint(2**31, (1,)).item()))

    # Pre-compute conditioning variable order based on global entropy
    # This ensures all branches at the same level use the same variable
    if cond == "naive_mle":
        # Sort variables by entropy (highest first)
        entropies = [(var, _compute_entropy(data, var, card_list[var])) for var in scope.query]
        entropies.sort(key=lambda x: -x[1])  # Descending by entropy
        variable_order = [var for var, _ in entropies]
    else:  # cond == "random"
        # Shuffle variables randomly
        perm = torch.randperm(len(scope.query), generator=rng)
        variable_order = [scope.query[i] for i in perm.tolist()]

    def _build_cnet(
        data_slice: Tensor,
        current_scope: Scope,
        var_order: list[int],  # Globally fixed order of conditioning variables
    ) -> Module:
        """Recursively build CNet structure.

        The structure is:
            Sum(  # OR-gate over conditioning variable values
                Product(Categorical(cond_var=v), SubCNet(remaining_scope)),
                ...
            )

        This ensures the full scope is preserved at each level.
        """
        n_instances = data_slice.shape[0]
        n_features = len(current_scope.query)

        # Base case: create leaf
        if n_features <= min_features_slice or n_instances < min_instances_slice:
            return _create_leaf(data_slice, current_scope, card_list, out_channels, alpha)

        # Find next conditioning variable from the pre-computed order
        cond_var = None
        for var in var_order:
            if var in current_scope.query:
                cond_var = var
                break

        if cond_var is None:
            # Fallback: no more variables to condition on
            return _create_leaf(data_slice, current_scope, card_list, out_channels, alpha)

        K = card_list[cond_var]
        remaining_scope = Scope([v for v in current_scope.query if v != cond_var])

        # If only one variable remains after conditioning, create leaf with full current scope
        if len(remaining_scope.query) == 0:
            return _create_leaf(data_slice, current_scope, card_list, out_channels, alpha)

        # Build branches for each value of the conditioning variable
        # Each branch is Product(Categorical(cond_var=v), SubCNet(remaining))
        branches: list[Module] = []
        weights_list: list[float] = []

        for v in range(K):
            mask = data_slice[:, cond_var] == v
            n_v = int(mask.sum().item())

            # Create deterministic Categorical for the conditioning variable
            # This will output P(cond_var = v) = 1 for this branch
            # Shape: (features=1, channels=out_channels, repetitions=1, K)
            cond_probs = torch.zeros(1, out_channels, 1, K)
            cond_probs[:, :, :, v] = 1.0
            cond_cat = Categorical(
                scope=Scope([cond_var]),
                out_channels=out_channels,
                num_repetitions=1,
                K=K,
                probs=cond_probs,
            )

            if n_v == 0:
                # Empty slice: create uniform leaf as fallback for remaining scope
                sub_cnet = _create_leaf(data_slice[:0], remaining_scope, card_list, out_channels, alpha)
            else:
                sub_cnet = _build_cnet(data_slice[mask], remaining_scope, var_order)

            # Combine cond_cat and sub_cnet with Product
            # Product over disjoint scopes
            branch = Product([cond_cat, sub_cnet])
            branches.append(branch)
            weights_list.append(float(n_v))

        # Normalize weights
        total = sum(weights_list)
        if total == 0:
            # Intentional statistical fallback for degenerate slices: keep branch mass uniform.
            weights_list = [1.0 / K] * K
        else:
            weights_list = [w / total for w in weights_list]

        # If only one branch has data, return it directly
        non_empty = [(i, w) for i, w in enumerate(weights_list) if w > 0]
        if len(non_empty) == 1:
            return branches[non_empty[0][0]]

        # Create Sum node over branches
        # Weight shape needs to account for each child's out_channels
        # Sum expects: (features, in_channels, out_channels, repetitions)
        # After Cat, in_channels = sum of all children's out_channels

        # Build weights properly - distribute each branch's weight across its channels
        weights_tensor_parts = []
        for idx, branch in enumerate(branches):
            branch_oc = branch.out_shape.channels
            branch_weight = weights_list[idx]
            # Each channel in this branch gets weight / branch_oc
            branch_weights = torch.full((branch_oc,), branch_weight / branch_oc)
            weights_tensor_parts.append(branch_weights)

        # Concatenate weights for all branches
        flat_weights = torch.cat(weights_tensor_parts)  # (total_in_channels,)

        # Shape: (features, in_channels, out_channels, repetitions)
        n_features_out = branches[0].out_shape.features
        total_in_channels = sum(b.out_shape.channels for b in branches)
        weights = flat_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, IC, 1, 1)
        weights = repeat(
            weights, "1 ic 1 1 -> f ic co 1", f=n_features_out, co=out_channels
        )  # (F, IC, OC, R)

        # Normalize weights to sum to 1 along in_channels dimension
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # Concatenate branches and create Sum
        cat_input = Cat(branches, dim=2)
        return Sum(inputs=cat_input, weights=weights)

    return _build_cnet(data, scope, variable_order)


def _create_leaf(
    data_slice: Tensor,
    scope: Scope,
    cardinalities: list[int],
    out_channels: int,
    alpha: float,
) -> Module:
    """Create a leaf node (CLTree or Categorical)."""
    n_features = len(scope.query)

    if n_features >= 2:
        # Use CLTree for multivariate scope
        K = max(cardinalities[v] for v in scope.query)
        leaf = CLTree(
            scope=scope,
            out_channels=out_channels,
            K=K,
            alpha=alpha,
        )
        if data_slice.shape[0] >= 2:
            leaf.maximum_likelihood_estimation(data_slice)
        return leaf
    else:
        # Use Categorical for single variable
        var = scope.query[0]
        K = cardinalities[var]
        leaf = Categorical(scope=scope, out_channels=out_channels, K=K)
        if data_slice.shape[0] >= 1:
            leaf.maximum_likelihood_estimation(data_slice)
        return leaf
