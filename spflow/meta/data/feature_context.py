"""Feature context management for probabilistic circuits.

This module provides the FeatureContext class, which tracks and manages the
probability distribution types for different features/variables in a dataset.
It maps feature indices to their corresponding distribution types (e.g., Gaussian,
Categorical, Bernoulli, etc.), enabling automatic leaf module selection during
structure and parameter learning.
"""
from inspect import isclass
from typing import Optional, Union

from spflow.meta.data.feature_types import FeatureType
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope


class FeatureContext:
    """Manages feature type specifications for probabilistic circuits.

    A FeatureContext maintains a mapping from feature indices (variables) to their
    probability distribution types. This enables the system to:
    - Automatically select appropriate leaf modules during structure learning
    - Handle heterogeneous data types (continuous, categorical, count data, etc.)
    - Validate consistency during model construction and learning

    Attributes:
        scope: The Scope object defining which features this context covers
        domain_map: Dictionary mapping feature indices to their distribution types
    """

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
        """Initialize a FeatureContext with feature type specifications.

        Args:
            scope: The Scope object defining which features this context covers
            domains: Feature type specifications. Can be:
                - None: Initialize all features with MetaType.Unknown
                - dict[int, type]: Map feature indices to their distribution types
                - list[type]: List of distribution types in scope order

        Raises:
            ValueError: If domains is a list with length != len(scope.query)
        """
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
        """Convert a FeatureType class to an instance if needed.

        Args:
            type: A FeatureType class or instance

        Returns:
            A FeatureType instance. If input was a class, returns a new instance.
            If input was already an instance, returns it unchanged.
        """
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
        """Set or update feature type specifications.

        Args:
            domains: Feature type specifications. Can be:
                - dict[int, type]: Maps feature indices to their distribution types
                - list[type]: List of distribution types (must match scope query length)
            overwrite: If True, allow replacing existing non-Unknown domain types.
                      If False (default), raise error if trying to replace existing types.

        Raises:
            ValueError: If domains list length doesn't match scope.query
            ValueError: If feature_id not in scope.query
            ValueError: If trying to overwrite existing domain without overwrite=True
        """
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
        """Retrieve feature type specifications for given features.

        Args:
            feature_ids: Feature indices to retrieve. Can be:
                - None: Return types for all features in scope.query
                - int: Return type for a single feature (returns single value)
                - list[int]: Return types for multiple features (returns list)

        Returns:
            The feature type(s) corresponding to the requested feature_id(s).
            Single int returns a single type, list returns a list of types.
        """
        if feature_ids is None:
            feature_ids = self.scope.query
        if isinstance(feature_ids, int):
            return self.domain_map[feature_ids]
        else:
            return [self.domain_map[feature_id] for feature_id in feature_ids]

    def select(self, feature_ids: Union[int, list[int]]) -> "FeatureContext":
        """Create a new FeatureContext for a subset of features.

        Useful for creating domain-specific FeatureContexts from a larger context.

        Args:
            feature_ids: Feature indices to include in the new context. Can be:
                - int: Single feature (will be converted to list)
                - list[int]: Multiple features

        Returns:
            A new FeatureContext with:
            - scope restricted to the selected features
            - domain_map containing only the selected feature types
            - evidence scope preserved from the original
        """
        if isinstance(feature_ids, int):
            feature_ids = [feature_ids]

        scope = Scope([int(f) for f in feature_ids], self.scope.evidence)
        domains = {feature_id: self.domain_map[feature_id] for feature_id in feature_ids}

        return FeatureContext(scope, domains)
