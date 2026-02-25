"""Shared builders for test suites."""

from __future__ import annotations

from spflow.meta import Scope
from spflow.modules.leaves.normal import Normal
from spflow.zoo.einet import Einet
from spflow.zoo.rat import RatSPN
from tests.utils.leaves import make_leaf


def make_rat_spn(
    *,
    leaf_cls,
    depth: int,
    n_region_nodes: int,
    num_leaves: int,
    num_repetitions: int,
    n_root_nodes: int,
    num_features: int,
    outer_product: bool,
    split_mode,
) -> RatSPN:
    leaf_layer = make_leaf(
        cls=leaf_cls,
        out_channels=num_leaves,
        out_features=num_features,
        num_repetitions=num_repetitions,
    )
    return RatSPN(
        leaf_modules=[leaf_layer],
        n_root_nodes=n_root_nodes,
        n_region_nodes=n_region_nodes,
        num_repetitions=num_repetitions,
        depth=depth,
        outer_product=outer_product,
        split_mode=split_mode,
    )


def make_einet_leaf_modules(num_features: int, num_leaves: int, num_repetitions: int) -> list[Normal]:
    return [
        Normal(
            scope=Scope([i]),
            out_channels=num_leaves,
            num_repetitions=num_repetitions,
        )
        for i in range(num_features)
    ]


def make_einet(
    *,
    num_features: int,
    num_classes: int,
    num_sums: int,
    num_leaves: int,
    depth: int,
    num_repetitions: int,
    layer_type: str = "einsum",
    structure: str = "top-down",
) -> Einet:
    return Einet(
        leaf_modules=make_einet_leaf_modules(num_features, num_leaves, num_repetitions),
        num_classes=num_classes,
        num_sums=num_sums,
        num_leaves=num_leaves,
        depth=depth,
        num_repetitions=num_repetitions,
        layer_type=layer_type,
        structure=structure,
    )
