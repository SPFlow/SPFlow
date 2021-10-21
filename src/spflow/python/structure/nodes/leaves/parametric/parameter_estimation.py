"""
Created on July 01, 2021

@authors: Bennet Wittelsbach
"""

import numpy as np
from typing import Any
from multipledispatch import dispatch  # type: ignore
from spflow.python.structure.nodes.leaves.parametric.exceptions import (
    InvalidParametersError,
    NotViableError,
)
from spflow.python.structure.nodes.leaves.parametric.parametric import (
    Gaussian,
    MultivariateGaussian,
    LogNormal,
    Bernoulli,
    Binomial,
    NegativeBinomial,
    Poisson,
    Geometric,
    Hypergeometric,
    Exponential,
    Gamma,
    get_scipy_object,
    get_scipy_object_parameters,
)
from spflow.python.structure.nodes.node import Node
from scipy.stats import lognorm, gamma  # type: ignore


# TODO: design decision: set mle params directly _in node_ or return them? first approach currently implemented
# TODO: design decision: _numpy arrays_ or default lists?

# TODO: update typing (see when numpy typing became available and if it collides with current requirements)
@dispatch(Node)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Node, data: Any) -> None:
    """Compute the parameters of the distribution represented by the node via MLE, if an closed-form estimator is available.

    Arguments:
        node:
            The node which parameters are to be estimated
        data:
            A 2-dimensional numpy-array holding the observations the parameters are to be estimated from.

    Raises:
        NotViableError:
            There is no (closed-form) maximum-likelihood estimator available for the type of distribution represented by node.
    """
    return NotViableError(f"There is no (closed-form) MLE for {node} implemented or existent")


@dispatch(Gaussian)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Gaussian, data: Any) -> None:
    data = validate_data(data, 1)
    node.mean = np.mean(data).item()
    node.stdev = np.std(data).item()

    if np.isclose(node.stdev, 0):
        node.stdev = 1e-8
    if get_scipy_object(node).pdf(node.mean, **get_scipy_object_parameters(node)) >= 1.0:
        print(f"Warning: 'Degenerated' PDF! Density at node.mean is greater than 1.0!")


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: MultivariateGaussian, data: Any) -> None:
    data = validate_data(data, data.shape[1])
    if data.shape[1] == 1:
        print(f"Warning: Trying to estimate MultivarateGaussian, but data has shape {data.shape}")

    node.mean_vector = np.mean(data, axis=0).tolist()
    node.covariance_matrix = np.cov(data, rowvar=0).tolist()

    # check for univariate degeneracy
    # for i in range(len(node.mean_vector)):
    #   [if pdf(mean) >= 1: print warning]

    pass


@dispatch(LogNormal)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: LogNormal, data: Any) -> None:
    data = validate_data(data, 1)

    # originally written by Alejandro Molina
    parameters = lognorm.fit(data, floc=0)
    node.mean = np.log(parameters[2]).item()
    node.stdev = parameters[0]


@dispatch(Bernoulli)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Bernoulli, data: Any) -> None:
    data = validate_data(data, 1)
    node.p = data.sum().item() / len(data)


@dispatch(Binomial)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Binomial, data: Any) -> None:
    data = validate_data(data, 1)
    node.n = len(data)
    node.p = data.sum().item() / (len(data) ** 2)


@dispatch(NegativeBinomial)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: NegativeBinomial, data: Any) -> None:
    raise NotViableError(
        "The Negative Binomal distribution parameters 'n, p' cannot be estimated via Maximum-Likelihood Estimation"
    )


@dispatch(Poisson)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Poisson, data: Any) -> None:
    data = validate_data(data, 1)
    node.l = np.mean(data).item()


@dispatch(Geometric)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Geometric, data: Any) -> None:
    data = validate_data(data, 1)
    node.p = len(data) / data.sum().item


@dispatch(Hypergeometric)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Hypergeometric, data: Any) -> None:
    raise NotViableError(
        "The Hypergeometric distribution parameters 'M, N, n' cannot be estimated via Maximum-Likelihood Estimation"
    )


@dispatch(Exponential)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Exponential, data: Any) -> None:
    data = validate_data(data, 1)
    node.l = np.mean(data).item()


@dispatch(Gamma)  # type: ignore[no-redef]
def maximum_likelihood_estimation(node: Gamma, data: Any) -> None:
    data = validate_data(data, 1)

    # default, originally written by Alejandro Molina
    node.alpha = 1.1
    node.beta = 1.0
    if np.any(data <= 0):
        # negative data? impossible gamma
        raise InvalidParametersError("All 'data' entries must not be 0")

    # zero variance? adding noise
    if np.isclose(np.std(data), 0):
        node.alpha = np.mean(data).item()
        print(f"Warning: {node} has 0 variance, adding noise")

    alpha, loc, theta = gamma.fit(data, floc=0)
    beta = 1.0 / theta
    if np.isfinite(alpha):
        node.alpha = alpha
        node.beta = beta
    else:
        raise InvalidParametersError(f"{node}: 'alpha' is not finite, parameters were NOT set")


def validate_data(data: Any, expected_dimensions: int, remove_nan: bool = True) -> Any:
    """Checking the data before using it for maximum-likelihood estimation.

    Arguments:
        data:
            A 2-dimensional numpy-array holding the data used for maximum-likelihood estimation
        expected_dimensions:
            The dimensions of the data and the distribution that is to be estimated. Equals 1 for all
            univariate distributions, else the number of dimensions of a multivariate distribution
        remove_nan:
            Boolean if nan entries (according to numpy.isnan()) shall be removed or kept.

    Returns:
        The validated and possibly cleaned up two-dimensional data numpy-array

    Raises:
        ValueError:
            If the shape of 'data' does not match 'expected' dimensions or is 0.
    """
    if data.shape[0] == 0 or data.shape[1] != expected_dimensions:
        raise ValueError(f"Argument 'data' must have shape of form (>0, {expected_dimensions}).")
    if np.isnan(data):
        print("Warning: Argument 'data' contains NaN values that are removed")
        if remove_nan:
            data = data[~np.isnan(data)]

    return data
