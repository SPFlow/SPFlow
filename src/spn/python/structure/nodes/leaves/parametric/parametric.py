"""
Created on June 11, 2021

@authors: Bennet Wittelsbach
"""

from abc import ABC, abstractmethod
from multipledispatch import dispatch  # type: ignore
from typing import Union, Dict, List, Tuple
from spn.python.structure.nodes import LeafNode, Node
from .exceptions import InvalidParametersError  # type: ignore
from .statistical_types import ParametricType  # type: ignore
from scipy.stats import (  # type: ignore
    norm,
    lognorm,
    multivariate_normal,
    uniform,
    bernoulli,
    binom,
    nbinom,
    poisson,
    geom,
    hypergeom,
    expon,
    gamma,
)
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete  # type: ignore
import numpy as np

# TODO:
#   - Categorical
#   - ... ?


class ParametricLeaf(LeafNode, ABC):
    """Base class for leaves that represent parametric probability distributions.

    Attributes:
        type:
            The parametric type of the distribution, either continuous or discrete

    """

    type: ParametricType

    def __init__(self, scope: List[int]) -> None:
        super().__init__(scope)


class Gaussian(ParametricLeaf):
    """(Univariate) Normal distribution

    PDF(x) =
        1/sqrt(2*pi*sigma^2) * exp(-(x-mu)^2/(2*sigma^2)), where
            - x is an observation
            - mu is the mean
            - sigma is the standard deviation

    Attributes:
        mean:
            mu
        stdev:
            sigma
    """

    type = ParametricType.CONTINUOUS

    def __init__(self, scope: List[int], mean: float, stdev: float) -> None:
        super().__init__(scope)
        self.set_params(mean, stdev)

    def set_params(self, mean: float, stdev: float) -> None:

        if not (np.isfinite(mean) and np.isfinite(stdev)):
            raise ValueError(
                f"Mean and standard deviation for Gaussian distribution must be finite, but were: {mean}, {stdev}"
            )
        if stdev <= 0.0:
            raise ValueError(
                f"Standard deviation for Gaussian distribution must be greater than 0.0, but was: {stdev}"
            )

        self.mean = mean
        self.stdev = stdev

    def get_params(self) -> Tuple[float, float]:
        return self.mean, self.stdev


class LogNormal(ParametricLeaf):
    """(Univariate) Log-Normal distribution

    PDF(x) =
        TODO

    Attributes:
        mean:
            mu
        stdev:
            sigma
    """

    type = ParametricType.POSITIVE

    def __init__(self, scope: List[int], mean: float, stdev: float) -> None:
        super().__init__(scope)
        self.set_params(mean, stdev)

    def set_params(self, mean: float, stdev: float) -> None:

        if not (np.isfinite(mean) and np.isfinite(stdev)):
            raise ValueError(
                f"Mean and standard deviation for LogNormal distribution must be finite, but were: {mean}, {stdev}"
            )
        if stdev <= 0.0:
            raise ValueError(
                f"Standard deviation for LogNormal distribution must be greater than 0.0, but was: {stdev}"
            )

        self.mean = mean
        self.stdev = stdev

    def get_params(self) -> Tuple[float, float]:
        return self.mean, self.stdev


class MultivariateGaussian(ParametricLeaf):
    """Multivariate Normal distribution

    PDF(x) =
        1/sqrt((2*pi)^d * det(cov)) * exp(-1/2 (x-mu)^T * cov * (x-mu)), where
            - d is the dimension of the distribution
            - x is the d-dim. vector of observations
            - mu is the d-dim. mean_vector
            - cov is the dxd covariance_matrix

    Attributes:
        mean_vector:
            A list holding the means (mu) of each of the one-dimensional Normal distributions.
            Has exactly as many elements as the scope of this leaf.
        covariance_matrix:
            A list of lists (representing a two-dimensional NxN matrix, where N is the length
            of the scope) describing the covariances of the distribution. The diagonal holds
            the variances (sigma^2) of each of the one-dimensional distributions.
    """

    type = ParametricType.CONTINUOUS

    def __init__(
        self,
        scope: List[int],
        mean_vector: List[float],
        covariance_matrix: List[List[float]],
    ) -> None:
        super().__init__(scope)
        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix

    def set_params(self, mean_vector: List[float], covariance_matrix: List[List[float]]) -> None:
        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix

    def get_params(self) -> Tuple[List[float], List[List[float]]]:
        return self.mean_vector, self.covariance_matrix


class Uniform(ParametricLeaf):
    """(Univariate) continuous Uniform distribution

    PDF(x) =
        1 / (end - start) * 1_[start, end], where
            - 1_[start, end] is the indicator function of the given interval (evaluating to 0 if x is not in the interval)


    Attributes:
        start:
            Start of interval. Must be less than beta.
        end:
            End of interval
    """

    type = ParametricType.CONTINUOUS

    def __init__(self, scope: List[int], start: float, end: float) -> None:
        super().__init__(scope)
        self.set_params(start, end)

    def set_params(self, start: float, end: float) -> None:

        if not start < end:
            raise ValueError(
                f"Lower bound for Uniform distribution must be less than upper bound, but were: {start}, {end}"
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"Lower and upper bound must be finite, but were: {start}, {end}")

        self.start = start
        self.end = end

    def get_params(self) -> Tuple[float, float]:
        return self.start, self.end


class Bernoulli(ParametricLeaf):
    """(Univariate) Binomial distribution

    PMF(k) =
        p   , if k=1
        1-p , if k=0

    Attributes:
        p:
            Probability of success (between 0.0 and 1.0)
    """

    type = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:
        super().__init__(scope)
        self.set_params(p)

    def set_params(self, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Bernoulli distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p = p

    def get_params(self) -> Tuple[float]:
        return (self.p,)


class Binomial(ParametricLeaf):
    """(Univariate) Binomial distribution

    PMF(k) =
        (n)C(k) * p^k * (1-p)^(n-k), where
            - (n)C(k) is the binomial coefficient (n choose k)

    Attributes:
        n:
            Number of i.i.d. Bernoulli trials (greater of equal to 0)
        p:
            Probability of success of each trial (between 0.0 and 1.0)
    """

    type = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: float) -> None:
        super().__init__(scope)

        self.set_params(n, p)

    def set_params(self, n: int, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Binomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Binomial distribution must to greater of equal to 0, but was: {n}"
            )

        self.n = n
        self.p = p

    def get_params(self) -> Tuple[int, float]:
        return self.n, self.p


class NegativeBinomial(ParametricLeaf):
    """(Univariate) Negative Binomial distribution

    PMF(k) =
        (k+n-1)C(k) * (1-p)^n * p^k, where
            - (n)C(k) is the binomial coefficient (n choose k)

    Attributes:
        n:
            Number of i.i.d. trials (greater of equal to 0)
        p:
            Probability of success of each trial (between 0.0 and 1.0)
    """

    type = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: float) -> None:
        super().__init__(scope)
        self.set_params(n, p)

    def set_params(self, n: int, p: float) -> None:

        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for NegativeBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for NegativeBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        self.n = n
        self.p = p

    def get_params(self) -> Tuple[int, float]:
        return self.n, self.p


class Poisson(ParametricLeaf):
    """(Univariate) Poisson distribution

    PMF(k) =
        l^k * exp(-l) / k!

    Attributes:
        l:
            Expected value (& variance) of the Poisson distribution (usually denoted as lambda)
    """

    type = ParametricType.COUNT

    def __init__(self, scope: List[int], l: float) -> None:
        super().__init__(scope)
        self.set_params(l)

    def set_params(self, l: float) -> None:

        if not np.isfinite(l):
            raise ValueError(f"Value of l for Poisson distribution must be finite, but was: {l}")

        self.l = l

    def get_params(self) -> Tuple[float]:
        return (self.l,)


class Geometric(ParametricLeaf):
    """(Univariate) Geometric distribution

    PMF(k) =
        p * (1-p)^(k-1)

    Attributes:
        p:
            Probability of success
    """

    type = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:
        super().__init__(scope)
        self.set_params(p)

    def set_params(self, p: float) -> None:

        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Geometric distribution must to be greater than 0.0 and less or equal to 1.0, but was: {p}"
            )

        self.p = p

    def get_params(self) -> Tuple[float]:
        return (self.p,)


class Hypergeometric(ParametricLeaf):
    """(Univariate) Hypergeometric distribution

    PMF(k) =
        (M)C(k) * (N-M)C(n-k) / (N)C(n), where
            - (n)C(k) is the binomial coefficient (n choose k)

    Attributes:
        N:
            Total number of entities (in the population)0, grater or equal to 0
        M:
            Number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N
        n:
            Number of observed entities (sample size), greater or equal to zero and less than or equal to N
    """

    type = ParametricType.COUNT

    def __init__(self, scope: List[int], N: int, M: int, n: int) -> None:
        super().__init__(scope)
        self.set_params(N, M, n)

    def set_params(self, N: int, M: int, n: int) -> None:

        if N < 0 or not np.isfinite(N):
            raise ValueError(
                f"Value of N for Hypergeometric distribution must be greater of equal to 0, but was: {N}"
            )
        if M < 0 or M > N or not np.isfinite(M):
            raise ValueError(
                f"Value of M for Hypergeometric distribution must be greater of equal to 0 and less or equal to N, but was: {M}"
            )
        if n < 0 or n > N or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Hypergeometric distribution must be greater of equal to 0 and less or equal to N, but was: {n}"
            )

        self.N = N
        self.M = M
        self.n = n

    def get_params(self) -> Tuple[int, int, int]:
        return self.N, self.M, self.n


class Exponential(ParametricLeaf):
    """(Univariate) Exponential distribution

    PDF(x) =
        l * exp(-l * x) , if x > 0
        0               , if x <= 0

    Attributes:
        l:
            TODO
    """

    type = ParametricType.POSITIVE

    def __init__(self, scope: List[int], l: float) -> None:
        super().__init__(scope)
        self.set_params(l)

    def set_params(self, l: float) -> None:

        if l <= 0.0 or not np.isfinite(l):
            raise ValueError(
                f"Value of l for Exponential distribution must be greater than 0, but was: {l}"
            )

        self.l = l

    def get_params(self) -> Tuple[float]:
        return (self.l,)


class Gamma(ParametricLeaf):
    """(Univariate) Gamma distribution

    PDF(x) =
        1/(G(beta) * alpha^beta) * x^(beta-1) * exp(-x/alpha)   , if x > 0
        0                                                       , if x <= 0, where
            - G(beta) is the Gamma function

    Attributes:
        alpha:
            Shape parameter, greater than 0
        beta:
            Scale parameter, greater than 0
    """

    type = ParametricType.POSITIVE

    def __init__(self, scope: List[int], alpha: float, beta: float) -> None:
        super().__init__(scope)
        self.set_params(alpha, beta)

    def set_params(self, alpha: float, beta: float) -> None:

        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(
                f"Value of alpha for Gamma distribution must be greater than 0, but was: {alpha}"
            )
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError(
                f"Value of beta for Gamma distribution must be greater than 0, but was: {beta}"
            )

        self.alpha = alpha
        self.beta = beta

    def get_params(self) -> Tuple[float, float]:
        return self.alpha, self.beta


@dispatch(Node)  # type: ignore[no-redef]
def get_scipy_object(node: Node) -> None:
    """Get the associated scipy object of a parametric leaf node. This can be used to call the PDF, CDF, PPF, etc.

    The standard implementation accepts nodes of any type and raises an error, if there is no scipy
    object implemented for the given node. Else, the respective dispatched function will be called
    which returns the associated scipy object.

    Arguments:
        node:
            The node of which the respective scipy object shall be returned

    Returns:
        A scipy object representing the distribution of the given node, or None.

    Raises:
        NotImplementedError:
            The node is a LeafNode and does not provide a scipy object or the node is not a LeafNode
            and cannot provide a scipy object.

    """
    if type(node) is LeafNode:
        raise NotImplementedError(f"{node} does not provide a scipy object")
    else:
        raise NotImplementedError(f"{node} cannot provide scipy objects")


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_scipy_object(node: Gaussian) -> rv_continuous:
    return norm


@dispatch(LogNormal)  # type: ignore[no-redef]
def get_scipy_object(node: LogNormal) -> rv_continuous:
    return lognorm


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def get_scipy_object(node: MultivariateGaussian) -> rv_continuous:
    return multivariate_normal


@dispatch(Uniform)  # type: ignore[no-redef]
def get_scipy_object(node: Uniform) -> rv_continuous:
    return uniform


@dispatch(Bernoulli)  # type: ignore[no-redef]
def get_scipy_object(node: Bernoulli) -> rv_discrete:
    return bernoulli


@dispatch(Binomial)  # type: ignore[no-redef]
def get_scipy_object(node: Binomial) -> rv_discrete:
    return binom


@dispatch(NegativeBinomial)  # type: ignore[no-redef]
def get_scipy_object(node: NegativeBinomial) -> rv_discrete:
    return nbinom


@dispatch(Poisson)  # type: ignore[no-redef]
def get_scipy_object(node: Poisson) -> rv_discrete:
    return poisson


@dispatch(Geometric)  # type: ignore[no-redef]
def get_scipy_object(node: Geometric) -> rv_discrete:
    return geom


@dispatch(Hypergeometric)  # type: ignore[no-redef]
def get_scipy_object(node: Hypergeometric) -> rv_discrete:
    return hypergeom


@dispatch(Exponential)  # type: ignore[no-redef]
def get_scipy_object(node: Exponential) -> rv_continuous:
    return expon


@dispatch(Gamma)  # type: ignore[no-redef]
def get_scipy_object(node: Gamma) -> rv_continuous:
    return gamma


@dispatch(Node)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Node) -> None:
    """Get the parameters of a paremetric leaf node, s.t. they can be directly passed to the PDF, CDF, etc. of the
    associated scipy object.

    The standard implementation accepts nodes of any type and raises an error, if it is a leaf node that does
    not provide parameters or the node is not a leaf node. Else, the respective dispatched function will be called
    which returns the associated parameters.

    Arguments:
        node:
            The node of which the parameters shall be returned

    Returns:
        A dictionary with {"parameter": value}, or None.

    Raises:
        NotImplementedError:
            The node is a LeafNode and does not provide parameters or the node is not a LeafNode.
        InvalidParametersError:
            The parameters are None or set to invalid values.

    """
    if type(node) is LeafNode:
        raise NotImplementedError(f"{node} does not provide any parameters")
    else:
        raise NotImplementedError(f"{node} cannot provide parameters")


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Gaussian) -> Dict[str, float]:
    if node.mean is None:
        raise InvalidParametersError(f"Parameter 'mean' of {node} must not be None")
    if node.stdev is None:
        raise InvalidParametersError(f"Parameter 'stdev' of {node} must not be None")
    parameters = {"loc": node.mean, "scale": node.stdev}
    return parameters


@dispatch(LogNormal)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: LogNormal) -> Dict[str, float]:
    if node.mean is None:
        raise InvalidParametersError(f"Parameter 'mean' of {node} must not be None")
    if node.stdev is None:
        raise InvalidParametersError(f"Parameter 'stdev' of {node} must not be None")

    parameters = {"loc": 0.0, "scale": np.exp(node.mean), "s": node.stdev}
    return parameters


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def get_scipy_object_parameters(
    node: MultivariateGaussian,
) -> Dict[str, Union[List[float], List[List[float]]]]:
    if node.mean_vector is None:
        raise InvalidParametersError(f"Parameter 'mean_vector' of {node} must not be None")
    if node.covariance_matrix is None:
        raise InvalidParametersError(f"Parameter 'covariance_matrix' of {node} must not be None")
    parameters = {"mean": node.mean_vector, "cov": node.covariance_matrix}
    return parameters


@dispatch(Uniform)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Uniform) -> Dict[str, float]:
    if node.start is None:
        raise InvalidParametersError(f"Parameter 'start' of {node} must not be None")
    if node.end is None:
        raise InvalidParametersError(f"Parameter 'end' of {node} must not be None")
    parameters = {"loc": node.start, "scale": node.end - node.start}
    return parameters


@dispatch(Bernoulli)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Bernoulli) -> Dict[str, float]:
    if node.p is None:
        raise InvalidParametersError(f"Parameter 'p' of {node} must not be None")
    parameters = {"p": node.p}
    return parameters


@dispatch(Binomial)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Binomial) -> Dict[str, Union[int, float]]:
    if node.n is None:
        raise InvalidParametersError(f"Parameter 'n' of {node} must not be None")
    if node.p is None:
        raise InvalidParametersError(f"Parameter 'p' of {node} must not be None")
    parameters = {"n": node.n, "p": node.p}
    return parameters


@dispatch(NegativeBinomial)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: NegativeBinomial) -> Dict[str, Union[int, float]]:
    if node.n is None:
        raise InvalidParametersError(f"Parameter 'n' of {node} must not be None")
    if node.p is None:
        raise InvalidParametersError(f"Parameter 'p' of {node} must not be None")
    parameters = {"n": node.n, "p": node.p}
    return parameters


@dispatch(Poisson)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Poisson) -> Dict[str, float]:
    if node.l is None:
        raise InvalidParametersError(f"Parameter 'l' of {node} must not be None")
    parameters = {"mu": node.l}
    return parameters


@dispatch(Geometric)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Geometric) -> Dict[str, float]:
    if node.p is None:
        raise InvalidParametersError(f"Parameter 'p' of {node} must not be None")
    parameters = {"p": node.p}
    return parameters


@dispatch(Hypergeometric)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Hypergeometric) -> Dict[str, int]:
    if node.N is None:
        raise InvalidParametersError(f"Parameter 'N' of {node} must not be None")
    if node.M is None:
        raise InvalidParametersError(f"Parameter 'M' of {node} must not be None")
    if node.n is None:
        raise InvalidParametersError(f"Parameter 'n' of {node} must not be None")
    # note: scipy hypergeom has switched semantics for the parameters
    parameters = {"M": node.N, "n": node.M, "N": node.n}
    return parameters


@dispatch(Exponential)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Exponential) -> Dict[str, float]:
    if node.l is None:
        raise InvalidParametersError(f"Parameter 'l' of {node} must not be None")
    parameters = {"scale": 1.0 / node.l}
    return parameters


@dispatch(Gamma)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Gamma) -> Dict[str, float]:
    if node.alpha is None:
        raise InvalidParametersError(f"Parameter 'alpha' of {node} must not be None")
    if node.beta is None:
        raise InvalidParametersError(f"Parameter 'beta' of {node} must not be None")
    parameters = {"a": node.alpha, "scale": 1.0 / node.beta}
    return parameters


if __name__ == "__main__":
    gauss_leaf = Gaussian(scope=[1], mean=0, stdev=1.0)
    # raise ValueError(get_scipy_object_parameters(gauss_leaf))
    print(
        get_scipy_object(gauss_leaf).pdf(
            x=[-1.0, 0, 1.0, 4.2], **get_scipy_object_parameters(gauss_leaf)
        )
    )
    print(
        get_scipy_object(gauss_leaf).cdf(
            x=[-1.0, 0, 1.0, 4.2], **get_scipy_object_parameters(gauss_leaf)
        )
    )
