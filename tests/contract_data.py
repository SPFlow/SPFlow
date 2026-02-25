"""Shared contract-level test matrices across non-leaf suites."""

from __future__ import annotations

from itertools import product

from spflow.modules import leaves
from spflow.modules.ops import SplitMode

# Zoo/RAT
RAT_DEPTH_VALUES = [1, 3]
RAT_REGION_NODES_VALUES = [1, 5]
RAT_NUM_LEAVES_VALUES = [1, 6]
RAT_NUM_REPETITIONS_VALUES = [1, 7]
RAT_ROOT_NODES_VALUES = [1, 4]
RAT_OUTER_PRODUCT_VALUES = [True, False]
RAT_SPLIT_MODE_VALUES = [None, SplitMode.consecutive(), SplitMode.interleaved()]
RAT_LEAF_CLS_VALUES = [
    leaves.Normal,
]
RAT_PARAMS = list(
    product(
        RAT_LEAF_CLS_VALUES,
        RAT_DEPTH_VALUES,
        RAT_REGION_NODES_VALUES,
        RAT_NUM_LEAVES_VALUES,
        RAT_NUM_REPETITIONS_VALUES,
        RAT_ROOT_NODES_VALUES,
        RAT_OUTER_PRODUCT_VALUES,
        RAT_SPLIT_MODE_VALUES,
    )
)
RAT_MULTI_DIST_PARAMS = list(
    product(
        RAT_REGION_NODES_VALUES,
        RAT_NUM_LEAVES_VALUES,
        RAT_NUM_REPETITIONS_VALUES,
        RAT_ROOT_NODES_VALUES,
        RAT_OUTER_PRODUCT_VALUES,
        RAT_SPLIT_MODE_VALUES,
    )
)

# Zoo/Einet
EINET_NUM_SUMS_VALUES = [3, 8]
EINET_NUM_LEAVES_VALUES = [3, 8]
EINET_DEPTH_VALUES = [0, 1, 2]
EINET_NUM_REPETITIONS_VALUES = [1, 3]
EINET_LAYER_TYPE_VALUES = ["einsum", "linsum"]
EINET_STRUCTURE_VALUES = ["top-down", "bottom-up"]
EINET_PARAMS_FULL = list(
    product(
        EINET_NUM_SUMS_VALUES,
        EINET_NUM_LEAVES_VALUES,
        EINET_DEPTH_VALUES,
        EINET_NUM_REPETITIONS_VALUES,
        EINET_LAYER_TYPE_VALUES,
        EINET_STRUCTURE_VALUES,
    )
)
EINET_PARAMS_SAMPLING = [p for p in EINET_PARAMS_FULL if p[5] == "top-down"]

# Learn/EM
EM_OUT_FEATURES_VALUES = [1, 4]
EM_OUT_CHANNELS_VALUES = [1, 3]
EM_NUM_REPETITION_VALUES = [1, 2]
EM_LEAF_CLS_VALUES = [
    leaves.Normal,
    leaves.Gamma,
    leaves.Poisson,
]
EM_PARAMS = list(
    product(
        EM_LEAF_CLS_VALUES,
        EM_OUT_FEATURES_VALUES,
        EM_OUT_CHANNELS_VALUES,
        EM_NUM_REPETITION_VALUES,
    )
)
