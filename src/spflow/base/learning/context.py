"""
@author Bennet Wittelsbach, based on code from Alejandro Molina
"""

from typing import List, Type
import numpy as np
from spflow.base.structure.nodes.leaves.parametric.parametric import ParametricLeaf

from spflow.base.structure.nodes.leaves.parametric.statistical_types import MetaType


class RandomVariableContext:
    """A RandomVariableContext provides meta information about data used to learn SPNs.

    In most cases, the following attributes need to be set by the user (if they're necessary for the type of SPN that shall be learned).
    E.g. parametric_types is only needed if the SPN has parametric distributions as leaves (e.g. Gaussian, Poisson, ...).

    Attributes:
        meta_types:
            List of meta types of the random variables in the data, indexed by the scope (index of the feature in the data).
        domains:
            Range of values of the associated random variable, indexed by the scope.
            Each domain is a numpy array containing either two values (real interval or binary RV) or discrete values (discrete RV).
        parametric_types:
            The type of the parametric distribution, if the random variables is represented by such one. Indexed by scope.
        feature_names:
            Name of the feature/random variable, indexed by scope. Optional.
    """

    def __init__(
        self,
        meta_types: List[int] = [],
        domains: List[np.ndarray] = [],
        parametric_types: List[Type[ParametricLeaf]] = [],
        feature_names: List[str] = [],
    ) -> None:
        # TODO: Do we need RandomVariableContext behind the scope of LearnSPN?
        #  Should RandomVariableContext be able to handle network_types?
        # TODO: E.g. should users define the network type of the learned PC here? And if so, extend it.
        self.meta_types = meta_types
        self.domains = domains
        self.parametric_types = parametric_types
        self.feature_names = feature_names

        if not meta_types and parametric_types:
            self.meta_types = []
            for p in parametric_types:
                self.meta_types.append(p.type.meta_type)

    def get_meta_types_by_scope(self, scopes: List[int]) -> List[MetaType]:
        """Get the MetaTypes of the random variables given by 'scopes'.

        Arguments:
            scopes:
                List of scopes of the random variables whose MetaType shall be retrieved.

        Returns:
            List of MetaTypes of all random variables given by 'scopes'.
        """
        return [MetaType(self.meta_types[s]) for s in scopes]

    def get_domains_by_scope(self, scopes: List[int]) -> List[np.ndarray]:
        """Get the domains of the random variables given by 'scopes'.

        Arguments:
            scopes:
                List of scopes of the random variables whose domains shall be retrieved.

        Returns:
            List of numpy arrays representing the domains of all random variables given by 'scopes'.
        """
        return [self.domains[s] for s in scopes]

    def get_parametric_types_by_scope(self, scopes) -> List[ParametricLeaf]:
        """Get the ParametricTypes of the random variables given by 'scopes'.

        Arguments:
            scopes:
                List of scopes of the random variables whose ParametricType shall be retrieved.

        Returns:
            List of ParametricType of all random variables given by 'scopes'.
        """
        return [self.parametric_types[s] for s in scopes]

    def add_domains(self, data: np.ndarray) -> "RandomVariableContext":
        """Infer the domains of each random variable from 2-dimensional data.

        The domain is the range of values between the minimum and the maximum value of each random variable present in 'data'.
        If the RV is discrete, the domain is an array of consecutive integers between the min and max value. If the RV is binary
        or continuous, the domain is an array of the min and max value only.

        Arguments:
            data:
                A (2-dimensional) numpy array from which the domains of each random variables are to be inferred.

        Returns:
            The RandomVariableContext itself.

        Raises:
            AssertionError:
                If the data is not 2-dimensional,
                OR the number of RV in the data does not match the number of MetaTypes,
                OR the MetaType is unknown (neither discrete nor binary nor continuous).
        """
        assert len(data.shape) == 2, "data is not 2D?"
        assert data.shape[1] == len(self.meta_types), "Data columns and metatype size doesn't match"

        from spflow.base.structure.nodes.leaves.parametric.statistical_types import MetaType

        domain = []

        for col in range(data.shape[1]):
            feature_meta_type = self.meta_types[col]
            min_val = np.nanmin(data[:, col])
            max_val = np.nanmax(data[:, col])
            domain_values = np.array([min_val, max_val])

            if feature_meta_type == MetaType.CONTINUOUS or feature_meta_type == MetaType.BINARY:
                domain.append(domain_values)
            elif feature_meta_type == MetaType.DISCRETE:
                domain.append(np.arange(domain_values[0], domain_values[1] + 1, 1))
            else:
                raise AssertionError("Unkown MetaType " + str(feature_meta_type))

        self.domains = domain  # np.asanyarray(domain)

        return self
