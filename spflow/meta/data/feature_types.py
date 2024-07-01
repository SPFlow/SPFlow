"""Feature types indicating the distribution or meta type of data features.
"""
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from spflow.meta.data.meta_type import MetaType


class FeatureType(ABC):
    """Abstract base class feature types.

    Attributes:
        meta_type:
            ``MetaType`` underlying the feature type.
    """

    meta_type: MetaType


# ----- continuous types -----
@dataclass
class ExponentialType(FeatureType):
    """Feature type for Exponential-distributed features.

    Attributes:
        rate:
            Floating point representing the rate parameter, greater than 0.0.
            Defaults to 1.0.
    """

    meta_type: ClassVar[MetaType] = MetaType.Continuous
    rate: float = 1.0


@dataclass
class GammaType(FeatureType):
    """Feature type for Gamma-distributed features.

    Attributes:
        alpha:
            Floating point representing the shape parameter, greater than 0.0.
            Defaults to 1.0.
        beta:
            Floating point representing the rate parameter, greater than 0.0.
            Defaults to 1.0.
    """

    meta_type: ClassVar[MetaType] = MetaType.Continuous
    alpha: float = 1.0
    beta: float = 1.0


@dataclass
class NormalType(FeatureType):
    """Feature type for Normal-distributed features.

    Attributes:
        mean:
            Floating point representing the mean.
            Defaults to 0.0.
        std:
            Floating point representing the standard deviation, greater than 0.0.
            Defaults to 1.0.
    """

    meta_type: ClassVar[MetaType] = MetaType.Continuous
    mean: float = 0.0
    std: float = 1.0


@dataclass
class LogNormalType(FeatureType):
    """Feature type for Log-Normal-distributed features.

    Attributes:
        mean:
            Floating point representing the mean.
            Defaults to 0.0.
        std:
            Floating point representing the standard deviation, greater than 0.0.
            Defaults to 1.0.
    """

    meta_type: ClassVar[MetaType] = MetaType.Continuous
    mean: float = 0.0
    std: float = 1.0


@dataclass
class UniformType(FeatureType):
    """Feature type for Uniform-distributed features.

    Attributes:
        start:
            Floating point representing the start of the interval (including).
        end:
            Floating point representing the end of the interval (including).
            Must be larger than ``start``.
    """

    meta_type: ClassVar[MetaType] = MetaType.Continuous
    start: float
    end: float


# ----- discrete types -----
@dataclass
class BernoulliType(FeatureType):
    """Feature type for Bernoulli-distributed features.

    Attributes:
        p:
            Success probability between 0 and 1.
            Defaults to 0.5.
    """

    meta_type: ClassVar[MetaType] = MetaType.Discrete
    p: float = 0.5


@dataclass
class BinomialType(FeatureType):
    """Feature type for Binomial-distributed features.

    Attributes:
        n:
            Number of i.i.d. Bernoulli trials.
        p:
            Success probability between 0 and 1.
            Defaults to 0.5.
    """

    meta_type: ClassVar[MetaType] = MetaType.Discrete
    n: int
    p: float = 0.5


@dataclass
class CategoricalType(FeatureType):
    """Feature type for Categorical-distributed features.

    Attributes:
        probs:
            List of probabilities for each category.
    """

    meta_type: ClassVar[MetaType] = MetaType.Discrete
    p: list[float]


@dataclass
class GeometricType(FeatureType):
    r"""Feature type for Geometric-distributed features.

    Attributes:
        p:
            Success probability in range :math:`(0,1]`.
            Defaults to 0.5.
    """
    meta_type: ClassVar[MetaType] = MetaType.Discrete
    p: float = 0.5


@dataclass
class HypergeometricType(FeatureType):
    """Feature type for Hypergeometric-distributed features.

    Attributes:
        N:
            Number of total entities in the population, greater or equal to 0.
        M:
            Number of entities of entries in the population, greater or equal to 0 and less than or equal to ``N``.
        n:
            Number of draws from the population, greater or equal to 0 and less than or equal to ``N``+``M``.
    """

    meta_type: ClassVar[MetaType] = MetaType.Discrete
    N: int
    M: int
    n: int


@dataclass
class NegativeBinomialType(FeatureType):
    r"""Feature type for Negative-Binomial-distributed features.

    Attributes:
        n:
            Number of total successes.
        p:
            Success probability in range :math:`(0,1]`.
            Defaults to 0.5.
    """
    meta_type: ClassVar[MetaType] = MetaType.Discrete
    n: int
    p: float = 0.5


@dataclass
class PoissonType(FeatureType):
    """Feature type for Poisson-distributed features.

    Attributes:
        rate:
            Floating point representing the rate parameter, greater than or equal to 0.0.
            Defaults to 1.0.
    """

    meta_type: ClassVar[MetaType] = MetaType.Discrete
    rate: float = 1.0


class FeatureTypes(ABC):
    r"""Abstract class keeping track of all registered feature types.

    Class is not meant to be instantiated and instead be interacted with directly.
    Members are all convenient accessors/aliases for ``MetaType`` enum members or ``FeatureType`` (sub-)classes.

    Members:
        Unknown:
            Alias for ``MetaType.Unknown``.
        Continuous:
            Alias for ``MetaType.Continuous``.
            Indicates continuous data, without any additional specification.
        Discrete:
            Alias for ``MetaType.Discrete``.
            Indicates discrete data, without any additional specification.
        Bernoulli:
            Alias for ``BernoulliType``, indicating Bernoulli-distributed data.
            Optional parameter ``p`` in :math:`[0,1]`, representing the success probability.
        Binomial:
            Alias for ``BinomialType``, indicating Binomial-distributed data.
            Required parameter ``n`` greater than or equal to 0, representing the number of i.i.d. Bernoulli trials.
            Optional parameter ``p`` in :math:`[0,1]`, representing the success probability.
        Exponential:
            Alias for ``ExponentialType``, indicating Exponential-distributed data.
            Optional parameter ``l``, representing the rate parameter, greater than 0.0.
        Gamma:
            Alias for ``GammaType``, indicating Gamma-distributed data.
            Optional parameters ``alpha`,``beta``, representing the shape and rate parameters, greater than 0.0.
            ``GammaType``
        Normal:
            Alias for ``NormalType``, indicating Normal-distributed data.
            Optional parameters ``mean``,``std``, representing the mean and standard deviation (the latter greater than 0.0).
        Geometric:
            Alias for ``GeometricType``, indicating Geometric-distributed data.
            Optional parameter ``p` in :math:`(0,1], representing the success probability.
        Hypergeometric:
            Alias for ``HypergeometricType``, indicating Hypergeometric-distributed data.
            Required parameters ``N``, ``M`` and ``n``.
            ``N`` represents the number of entities in the population, greater than or equal to 0.
            ``M`` represents the number of entities of interest in the population, greater than or equal to zero and less than or equal to ``N``.
            ``n`` represents the number of draws, greater than or equal to zero and less than or equal to ``N``.
        Log-Normal:
            Alias for ``LogNormalType``, indicating Log-Normal-distributed data.
            Optional parameters ``mean``,``std``, representing the mean and standard deviation (the latter greater than 0.0).
        NegativeBinomial:
            Required parameter ``n`` greater than or equal to 0, representing the number of total successes.
            Optional parameter ``p`` in :math:`(0,1]`, representing the success probability.
        Poisson:
            Alias for ``PoissonType``, indicating Poisson-distributed data.
            Optional parameter ``l``, representing the rate parameter, greater than or equal to 0.0.
        Uniform:
            Alias for ``UniformType``, indicating Uniform-distributed data.
            Required parameters ``start``,``end``, representing the start (including) and end (including) of the interval, the latter greater than ``start``.
    """
    # ----- meta feature types -----
    Unknown = MetaType.Unknown
    Continuous = MetaType.Continuous
    Discrete = MetaType.Discrete

    # ----- continuous feature types -----
    Exponential = ExponentialType
    Gamma = GammaType
    Normal = NormalType
    LogNormal = LogNormalType
    Uniform = UniformType

    # ----- discrete feature types -----
    Bernoulli = BernoulliType
    Binomial = BinomialType
    Geometric = GeometricType
    Hypergeometric = HypergeometricType
    NegativeBinomial = NegativeBinomialType
    Poisson = PoissonType
    Categorical = CategoricalType

    @classmethod
    def register(cls, name: str, type: FeatureType, overwrite=False) -> None:
        """Registers a feature type.

        Args:
            name:
                String specifying the name the feature type should be registered under.
            type:
                ``FeatureType`` (sub-)class or instance thereof.
            overwrite:
                Boolean indicating whether or not to overwrite any potentially existing feature type registered under the same name.
                Defaults to False.
        """
        if hasattr(cls, name) and not overwrite:
            raise ValueError(
                "Feature type {name} is already registered. If type should be overwritten, enable 'overwrite'."
            )

        setattr(cls, name, type)
