"""Shared parameter grids for non-leaf module contract tests."""

from __future__ import annotations

from itertools import product

IN_CHANNELS_VALUES = [1, 4]
OUT_CHANNELS_VALUES = [1, 5]
OUT_FEATURES_VALUES = [1, 6]
NUM_REPETITIONS_VALUES = [1, 7]

SUM_PARAMS = list(
    product(
        IN_CHANNELS_VALUES,
        OUT_CHANNELS_VALUES,
        OUT_FEATURES_VALUES,
        NUM_REPETITIONS_VALUES,
    )
)

PRODUCT_PARAMS = list(product([1, 3], [1, 4], [1, 5]))

CAT_PARAMS = list(
    product(
        OUT_CHANNELS_VALUES,
        OUT_FEATURES_VALUES,
        [1, 5],
        [1, 2],
    )
)

SPLIT_PARAMS = list(
    product(
        OUT_CHANNELS_VALUES,
        [3, 8],
        [2, 4],
        NUM_REPETITIONS_VALUES,
    )
)

CONV_HW_VALUES = [(4, 4), (8, 8)]
CONV_KERNEL_VALUES = [2]
