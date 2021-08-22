"""
Created on April 15, 2018
@author: Alejandro Molina

Modified on June 26, 2021 by Bennet Wittelsbach
"""
from enum import Enum


class MetaType(Enum):
    CONTINUOUS = 1
    BINARY = 2
    DISCRETE = 3


class ParametricType(Enum):
    CONTINUOUS = (1, MetaType.CONTINUOUS)
    INTERVAL = (2, MetaType.CONTINUOUS)
    POSITIVE = (3, MetaType.CONTINUOUS)
    CATEGORICAL = (4, MetaType.DISCRETE)
    ORDINAL = (5, MetaType.DISCRETE)
    COUNT = (6, MetaType.DISCRETE)
    BINARY = (7, MetaType.BINARY)

    def __init__(self, enum_val: int, meta_type: int):
        self._enum_val = enum_val
        self._meta_type = meta_type

    @property
    def meta_type(self) -> int:
        return self._meta_type


META_TYPE_MAP = {
    MetaType.CONTINUOUS: [
        ParametricType.CONTINUOUS,
        ParametricType.INTERVAL,
        ParametricType.POSITIVE,
    ],
    MetaType.BINARY: [ParametricType.BINARY],
    MetaType.DISCRETE: [
        ParametricType.CATEGORICAL,
        ParametricType.ORDINAL,
        ParametricType.COUNT,
    ],
}
