"""TODO
"""
from inspect import isclass
from typing import Optional, Union

from spflow.meta.data.feature_types import FeatureType
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope


class FeatureContext:
    """TODO"""

    def __init__(
        self,
        scope: Scope,
        domains: Optional[
            Union[
                dict[int, Union[MetaType, FeatureType, type[FeatureType]]],
                list[Union[MetaType, FeatureType, type[FeatureType]]],
            ]
        ] = None,
    ):
        """TODO"""
        if domains is None:
            domains = {}

        # store scope
        self.scope = scope

        # initialize map
        self.domain_map = {feature_id: MetaType.Unknown for feature_id in scope.query}

        # parse and add type contexts
        self.set_domains(domains)

    @classmethod
    def parse_type(cls, type: FeatureType):
        """TODO"""
        if isclass(type):
            type = type()

        return type

    def set_domains(
        self,
        domains: Union[
            dict[int, Union[MetaType, FeatureType, type[FeatureType]]],
            list[Union[MetaType, FeatureType, type[FeatureType]]],
        ],
        overwrite: bool = False,
    ):
        """TODO"""
        if isinstance(domains, list):
            if len(domains) != len(self.scope.query):
                raise ValueError(
                    "Length of list of domain types for 'FeatureContext' does not match number of scope query RVs."
                )

            domains = {
                feature_id: feature_type for feature_id, feature_type in zip(self.scope.query, domains)
            }

        for feature_id, feature_type in domains.items():
            # convert to instance if necessary
            feature_type = self.parse_type(feature_type)

            if feature_id not in self.scope.query:
                raise ValueError(f"Feature index {feature_id} is not part of the query scope.")

            if self.domain_map[feature_id] != MetaType.Unknown and not overwrite:
                raise ValueError(
                    f"Domain for feature index {feature_id} is already specified (i.e., not 'MetaType.Unknown'). If domain should be overwritten, enable 'overwrite'."
                )

            self.domain_map[feature_id] = feature_type

    def get_domains(
        self, feature_ids: Union[int, list[int]] = None
    ) -> Union[
        Union[MetaType, FeatureType, type[FeatureType]],
        list[Union[MetaType, FeatureType, type[FeatureType]]],
    ]:
        """TODO"""
        if feature_ids is None:
            feature_ids = self.scope.query
        if isinstance(feature_ids, int):
            return self.domain_map[feature_ids]
        else:
            return [self.domain_map[feature_id] for feature_id in feature_ids]

    def select(self, feature_ids: Union[int, list[int]]) -> "FeatureContext":
        """TODO"""
        if isinstance(feature_ids, int):
            feature_ids = [feature_ids]

        scope = Scope([int(f) for f in feature_ids], self.scope.evidence)
        domains = {feature_id: self.domain_map[feature_id] for feature_id in feature_ids}

        return FeatureContext(scope, domains)
