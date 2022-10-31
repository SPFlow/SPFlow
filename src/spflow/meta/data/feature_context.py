# -*- coding: utf-8 -*-
"""TODO
"""
"""
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.feature_types import FeatureType, FeatureTypes

from typing import Optional, Union, Iterable, Dict, Literal
from inspect import isclass


class FeatureContext:
    def __init__(self, domains: Optional[Union[
                                Dict[int, Union[MetaType, FeatureType]],
                                Iterable[Union[MetaType, FeatureType]]
                            ]]=None):
    
        if domains is None:
            domains = {}

        # initialize map
        self.domain_map = {}

        # parse and add type contexts
        self.add(domains)

    @classmethod
    def parse_type(self, type: FeatureType):
        if isclass(type):
            type = type()
    
        return type
    
    def add(self,
        domains: Union[
            Dict[int, Union[MetaType, FeatureType]],
            Iterable[Union[MetaType, FeatureType]]
        ],
        overwrite: bool=True):

        if not isinstance(domains, Dict):
            domains = {i: t for i, t in enumerate(domains)}

        for feature_id, feature_type in domains.items():
            # convert to instance if necessary
            feature_type = self.parse_type(feature_type)

            if feature_id in self.domain_map and not overwrite:
                raise ValueError("Context for data index {feature_id} is already specified. If context should be overwritten, enable 'overwrite'.")

            self.map[feature_id] = feature_type
"""