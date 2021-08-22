"""
Created on July 4, 2021

@authors: Philipp Deibert 
"""

from abc import ABC
from typing import List, Union, Tuple

from multipledispatch import dispatch  # type: ignore

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spn.torch.structure.nodes.node import TorchLeafNode
from spn.python.structure.nodes.leaves.parametric.statistical_types import ParametricType
import spn.python.structure.nodes.leaves.parametric.parametric as P


class TorchParametricLeaf(TorchLeafNode, ABC):
    """Base class for Torch leaf nodes representing parametric probability distributions.

    Attributes:
        type (ParametricType): The parametric type of the distribution, either continuous or discrete.
    """

    ptype: ParametricType

    def __init__(self, scope: List[int]) -> None:
        super(TorchParametricLeaf, self).__init__(scope)


class TorchGaussian(TorchParametricLeaf):
    """(Univariate) Normal distribution.

    PDF(x) =
        1/sqrt(2*pi*sigma^2) * exp(-(x-mu)^2/(2*sigma^2)), where
            - x is an observation
            - mu is the mean
            - sigma is the standard deviation

    Attributes:
        mean (float): mean (mu) of the distribution.
        stdev (float): standard deviation (sigma) of the distribution.
    """

    ptype = ParametricType.CONTINUOUS

    def __init__(self, scope: List[int], mean: float, stdev: float) -> None:
        super(TorchGaussian, self).__init__(scope)

        if len(scope) != 1:
            raise ValueError("Invalid scope size for univariate Gaussian")

        # register mean and standard deviation as torch parameters
        self.register_parameter("mean", Parameter(torch.tensor(float(mean))))
        self.register_parameter("stdev", Parameter(torch.tensor(float(stdev))))

        # create Torch distribution with specified parameters
        self.dist = D.Normal(loc=self.mean, scale=self.stdev)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, mean: float, stdev: float) -> None:
        self.mean.data = torch.tensor(mean)
        self.stdev.data = torch.tensor(stdev)

    def get_params(self) -> Tuple[float, float]:
        return self.mean.data.cpu().numpy(), self.stdev.data.cpu().numpy()  # type: ignore


@dispatch(P.Gaussian)  # type: ignore[no-redef]
def toTorch(node: P.Gaussian) -> TorchGaussian:
    return TorchGaussian(node.scope, *node.get_params())


@dispatch(TorchGaussian)  # type: ignore[no-redef]
def toNodes(torch_node: TorchGaussian) -> P.Gaussian:
    return P.Gaussian(torch_node.scope, *torch_node.get_params())


class TorchLogNormal(TorchParametricLeaf):
    """(Univariate) Log-Normal distribution.

    Attributes:
        mean (float): mean (mu) of the distribution.
        stdev (float): standard deviation (sigma) of the distribution.
    """

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], mean: float, stdev: float) -> None:
        super(TorchLogNormal, self).__init__(scope)

        # register mean and standard deviation as torch parameters
        self.register_parameter("mean", Parameter(torch.tensor(mean, dtype=torch.float32)))
        self.register_parameter("stdev", Parameter(torch.tensor(stdev, dtype=torch.float32)))

        # create Torch distribution with specified parameters
        self.dist = D.LogNormal(loc=self.mean, scale=self.stdev)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, mean: float, stdev: float) -> None:
        self.mean.data = torch.tensor(mean)
        self.stdev.data = torch.tensor(stdev)

    def get_params(self) -> Tuple[float, float]:
        return self.mean.data.cpu().numpy(), self.stdev.data.cpu().numpy()  # type: ignore


@dispatch(P.LogNormal)  # type: ignore[no-redef]
def toTorch(node: P.LogNormal) -> TorchLogNormal:
    return TorchLogNormal(node.scope, *node.get_params())


@dispatch(TorchLogNormal)  # type: ignore[no-redef]
def toNodes(torch_node: TorchLogNormal) -> P.LogNormal:
    return P.LogNormal(torch_node.scope, *torch_node.get_params())


class TorchMultivariateGaussian(TorchParametricLeaf):
    """Multivariate Normal distribution.

    PDF(x) =
        1/sqrt((2*pi)^d * det(cov)) * exp(-1/2 (x-mu)^T * cov * (x-mu)), where
            - d is the dimension of the distribution
            - x is the d-dim. vector of observations
            - mu is the d-dim. mean_vector
            - cov is the dxd covariance_matrix

    Attributes:
        mean_vector (Union[List[float], torch.Tensor, np.ndarray]): A list, NumPy array or a PyTorch tensor holding the means (mu) for each dimension of the distribution (has exactly as mUnion[int, float] elements as the scope of this leaf).
        covariance_matrix: A list of lists (representing a two-dimensional NxN matrix, where N is the length
            of the scope) describing the covariances of the distribution. The diagonal holds
            the variances (sigma^2) of each of the one-dimensional distributions.
    """

    ptype = ParametricType.POSITIVE

    def __init__(
        self,
        scope: List[int],
        mean_vector: Union[List[float], torch.Tensor, np.ndarray],
        covariance_matrix: Union[List[List[float]], torch.Tensor, np.ndarray],
    ) -> None:
        super(TorchMultivariateGaussian, self).__init__(scope)

        if isinstance(mean_vector, list):
            # convert float list to torch tensor
            mean_vector = torch.tensor(mean_vector, dtype=torch.float32)
        elif isinstance(mean_vector, np.ndarray):
            # convert numpy array to torch tensor
            mean_vector = torch.from_numpy(mean_vector)

        if isinstance(covariance_matrix, list):
            # convert numpy array to torch tensor
            covariance_matrix = torch.tensor(covariance_matrix, dtype=torch.float32)
        elif isinstance(covariance_matrix, np.ndarray):
            # convert numpy array to torch tensor
            covariance_matrix = torch.from_numpy(covariance_matrix)

        # register mean and covariance as torch parameters
        self.register_parameter("mean_vector", Parameter(mean_vector))
        self.register_parameter("covariance_matrix", Parameter(covariance_matrix))

        # create Torch distribution with specified parameters
        self.dist = D.MultivariateNormal(
            loc=self.mean_vector, covariance_matrix=self.covariance_matrix
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(
        self,
        mean_vector: Union[List[float], torch.Tensor, np.ndarray],
        covariance_matrix: Union[List[List[float]], torch.Tensor, np.ndarray],
    ) -> None:

        if isinstance(mean_vector, list):
            # convert float list to torch tensor
            mean_vector = torch.tensor(mean_vector, dtype=torch.float32)
        elif isinstance(mean_vector, np.ndarray):
            # convert numpy array to torch tensor
            mean_vector = torch.from_numpy(mean_vector)

        if isinstance(covariance_matrix, list):
            # convert numpy array to torch tensor
            covariance_matrix = torch.tensor(covariance_matrix, dtype=torch.float32)
        elif isinstance(covariance_matrix, np.ndarray):
            # convert numpy array to torch tensor
            covariance_matrix = torch.from_numpy(covariance_matrix)

        self.mean_vector.data = mean_vector
        self.covariance_matrix.data = covariance_matrix

    def get_params(self) -> Tuple[List[float], List[List[float]]]:
        return self.mean_vector.data.cpu().tolist(), self.covariance_matrix.data.cpu().tolist()  # type: ignore


@dispatch(P.MultivariateGaussian)  # type: ignore[no-redef]
def toTorch(node: P.MultivariateGaussian) -> TorchMultivariateGaussian:
    return TorchMultivariateGaussian(node.scope, *node.get_params())


@dispatch(TorchMultivariateGaussian)  # type: ignore[no-redef]
def toNodes(torch_node: TorchMultivariateGaussian) -> P.MultivariateGaussian:
    return P.MultivariateGaussian(torch_node.scope, *torch_node.get_params())


class TorchUniform(TorchParametricLeaf):
    """(Univariate) continuous Uniform distribution.

    PDF(x) =
        1 / (end - start) * 1_[start, end], where
            - 1_[start, end] is the indicator function of the given interval (evaluating to 0 if x is not in the interval)


    Attributes:
        start (float): Start of the interval. Must be less than beta.
        end (float): End of interval.
    """

    ptype = ParametricType.CONTINUOUS

    def __init__(self, scope: List[int], start: float, end: float) -> None:
        super(TorchUniform, self).__init__(scope)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.tensor(start, dtype=torch.float32))
        self.register_buffer("end", torch.tensor(end, dtype=torch.float32))

        # create Torch distribution with specified parameters
        self.dist = D.Uniform(low=self.start, high=self.end)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, start: float, end: float) -> None:
        self.start.data = torch.tensor(start)  # type: ignore
        self.end.data = torch.tensor(end)  # type: ignore

    def get_params(self) -> Tuple[float, float]:
        return self.start.cpu().numpy(), self.end.cpu().numpy()  # type: ignore


@dispatch(P.Uniform)  # type: ignore[no-redef]
def toTorch(node: P.Uniform) -> TorchUniform:
    return TorchUniform(node.scope, node.start, node.end)


@dispatch(TorchUniform)  # type: ignore[no-redef]
def toNodes(torch_node: TorchUniform) -> P.Uniform:
    return P.Uniform(torch_node.scope, torch_node.start.cpu().numpy(), torch_node.end.cpu().numpy())  # type: ignore


class TorchBernoulli(TorchParametricLeaf):
    """(Univariate) Binomial distribution.

    PMF(k) =
        p   , if k=1
        1-p , if k=0

    Attributes:
        p (float): Probability of success.
    """

    ptype = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:
        super(TorchBernoulli, self).__init__(scope)

        # register success probability p as torch parameter
        self.register_parameter("p", Parameter(torch.tensor(p, dtype=torch.float32)))

        # create Torch distribution with specified parameters
        self.dist = D.Bernoulli(probs=self.p)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, p: float) -> None:
        self.p.data = torch.tensor(p, dtype=torch.float32)

    def get_params(self) -> Tuple[float]:
        return (self.p.data.cpu().numpy(),)  # type: ignore


@dispatch(P.Bernoulli)  # type: ignore[no-redef]
def toTorch(node: P.Bernoulli) -> TorchBernoulli:
    return TorchBernoulli(node.scope, *node.get_params())


@dispatch(TorchBernoulli)  # type: ignore[no-redef]
def toNodes(torch_node: TorchBernoulli) -> P.Bernoulli:
    return P.Bernoulli(torch_node.scope, *torch_node.get_params())


class TorchBinomial(TorchParametricLeaf):
    """(Univariate) Binomial distribution.

    PMF(k) =
        (n)C(k) * p^k * (1-p)^(n-k), where
            - (n)C(k) is the binomial coefficient (n choose k)

    Attributes:
        n (int): Number of i.i.d. Bernoulli trials.
        p (float): Probability of success of each trial.
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: float) -> None:
        super(TorchBinomial, self).__init__(scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.tensor(n))

        # register success probability p as torch parameter
        self.register_parameter("p", Parameter(torch.tensor(p, dtype=torch.float32)))

        # create Torch distribution with specified parameters
        self.dist = D.Binomial(total_count=self.n, probs=self.p)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, n: int, p: float) -> None:
        self.n.data = torch.tensor(n)  # type: ignore
        self.p.data = torch.tensor(p)  # type: ignore

    def get_params(self) -> Tuple[int, float]:
        return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore


@dispatch(P.Binomial)  # type: ignore[no-redef]
def toTorch(node: P.Binomial) -> TorchBinomial:
    return TorchBinomial(node.scope, *node.get_params())


@dispatch(TorchBinomial)  # type: ignore[no-redef]
def toNodes(torch_node: TorchBinomial) -> P.Binomial:
    return P.Binomial(torch_node.scope, *torch_node.get_params())


class TorchNegativeBinomial(TorchParametricLeaf):
    """(Univariate) Negative Binomial distribution.

    PMF(k) =
        (k+n-1)C(k) * (1-p)^n * p^k, where
            - (n)C(k) is the binomial coefficient (n choose k)

    Attributes:
        n (int): Number of i.i.d. trials.
        p (float): Probability of success of each trial.
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: float) -> None:
        super(TorchNegativeBinomial, self).__init__(scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.tensor(n))

        # register success probability p as torch parameter
        self.register_parameter("p", Parameter(torch.tensor(p, dtype=torch.float32)))

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        # note: the distribution is not stored as an attribute due to mismatching parameters after gradient updates (gradients don't flow back to p when initializing with 1.0-p)
        dist = D.NegativeBinomial(total_count=self.n, probs=torch.ones(1) - self.p)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, n: int, p: float) -> None:
        self.n.data = torch.tensor(n)  # type: ignore
        self.p.data = torch.tensor(p)  # type: ignore

    def get_params(self) -> Tuple[int, float]:
        return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore


@dispatch(P.NegativeBinomial)  # type: ignore[no-redef]
def toTorch(node: P.NegativeBinomial) -> TorchNegativeBinomial:
    return TorchNegativeBinomial(node.scope, *node.get_params())


@dispatch(TorchNegativeBinomial)  # type: ignore[no-redef]
def toNodes(torch_node: TorchNegativeBinomial) -> P.NegativeBinomial:
    return P.NegativeBinomial(torch_node.scope, *torch_node.get_params())


class TorchPoisson(TorchParametricLeaf):
    """(Univariate) Poisson distribution.

    PMF(k) =
        l^k * exp(-l) / k!

    Attributes:
        l (float): Expected value (& variance) of the Poisson distribution (usually denoted as lambda).
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], l: float) -> None:
        super(TorchPoisson, self).__init__(scope)

        # register lambda l as torch parameter
        self.register_parameter("l", Parameter(torch.tensor(l, dtype=torch.float32)))

        # create Torch distribution with specified parameters
        self.dist = D.Poisson(rate=self.l)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, l: float) -> None:
        self.l.data = torch.tensor(l)

    def get_params(self) -> Tuple[float]:
        return (self.l.data.cpu().numpy(),)  # type: ignore


@dispatch(P.Poisson)  # type: ignore[no-redef]
def toTorch(node: P.Poisson) -> TorchPoisson:
    return TorchPoisson(node.scope, *node.get_params())


@dispatch(TorchPoisson)  # type: ignore[no-redef]
def toNodes(torch_node: TorchPoisson) -> P.Poisson:
    return P.Poisson(torch_node.scope, *torch_node.get_params())


class TorchGeometric(TorchParametricLeaf):
    """(Univariate) Geometric distribution.

    PMF(k) =
        p * (1-p)^(k-1)

    Attributes:
        p (float): Probability of success.
    """

    ptype = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:
        super(TorchGeometric, self).__init__(scope)

        # register success probability p as torch parameter
        self.register_parameter("p", Parameter(torch.tensor(p, dtype=torch.float32)))

        # create Torch distribution with specified parameters
        self.dist = D.Geometric(probs=self.p)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask] - 1)

        return log_prob

    def set_params(self, p: float) -> None:
        self.p.data = torch.tensor(p)

    def get_params(self) -> Tuple[float]:
        return (self.p.data.cpu().numpy(),)  # type: ignore


@dispatch(P.Geometric)  # type: ignore[no-redef]
def toTorch(node: P.Geometric) -> TorchGeometric:
    return TorchGeometric(node.scope, *node.get_params())


@dispatch(TorchGeometric)  # type: ignore[no-redef]
def toNodes(torch_node: TorchGeometric) -> P.Geometric:
    return P.Geometric(torch_node.scope, *torch_node.get_params())


class TorchHypergeometric(TorchParametricLeaf):
    """(Univariate) Hypergeometric distribution.

    PMF(k) =
        (M)C(k) * (N-M)C(n-k) / (N)C(n), where
            - (n)C(k) is the binomial coefficient (n choose k)

    Attributes:
        N (int): Total number of entities (in the population).
        M (int): Number of entities with property of interest (in the population), less than or equal to N.
        n (int): Number of observed entities (sample size), less than or equal to N.
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], N: int, M: int, n: int) -> None:
        super(TorchHypergeometric, self).__init__(scope)

        self.N = N
        self.M = M
        self.n = n

        # register parameters as torch buffers (should not be changed)
        self.register_buffer("N", torch.tensor(N, dtype=torch.int32))
        self.register_buffer("M", torch.tensor(M, dtype=torch.int32))
        self.register_buffer("n", torch.tensor(n, dtype=torch.int32))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Distribution is not yet supported.")

    def set_params(self, N: int, M: int, n: int) -> None:
        self.N = N
        self.M = M
        self.n = n

        # self.N = torch.tensor(N, dtype=torch.int32)
        # self.M = torch.tensor(M, dtype=torch.int32)
        # self.n = torch.tensor(n, dtype=torch.int32)

        # TODO buffer and attribute to same tensor

    def get_params(self) -> Tuple[int, int, int]:
        # TODO: buffer
        return self.N.data.cpu().numpy(), self.M.data.cpu().numpy(), self.n.data.cpu().numpy()  # type: ignore


@dispatch(P.Hypergeometric)  # type: ignore[no-redef]
def toTorch(node: P.Hypergeometric) -> TorchHypergeometric:
    return TorchHypergeometric(node.scope, *node.get_params())


@dispatch(TorchHypergeometric)  # type: ignore[no-redef]
def toNodes(torch_node: TorchHypergeometric) -> P.Hypergeometric:
    return P.Hypergeometric(torch_node.scope, *torch_node.get_params())


class TorchExponential(TorchParametricLeaf):
    """(Univariate) Exponential distribution.

    PDF(x) =
        l * exp(-l * x) , if x > 0
        0               , if x <= 0

    Attributes:
        l (float): TODO
    """

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], l: float) -> None:
        super(TorchExponential, self).__init__(scope)

        # TODO
        self.register_parameter("l", Parameter(torch.tensor(l, dtype=torch.float32)))

        # create Torch distribution with specified parameters
        self.dist = D.Exponential(rate=self.l)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, l: float) -> None:
        self.l.data = torch.tensor(l)

    def get_params(self) -> Tuple[float]:
        return (self.l.data.cpu().numpy(),)  # type: ignore


@dispatch(P.Exponential)  # type: ignore[no-redef]
def toTorch(node: P.Exponential) -> TorchExponential:
    return TorchExponential(node.scope, *node.get_params())


@dispatch(TorchExponential)  # type: ignore[no-redef]
def toNodes(torch_node: TorchExponential) -> P.Exponential:
    return P.Exponential(torch_node.scope, *torch_node.get_params())


class TorchGamma(TorchParametricLeaf):
    """(Univariate) Gamma distribution.

    PDF(x) =
        1/(G(beta) * alpha^beta) * x^(beta-1) * exp(-x/alpha)   , if x > 0
        0                                                       , if x <= 0, where
            - G(beta) is the Gamma function

    Attributes:
        alpha (float): Shape parameter, greater than 0.
        beta (float): Scale parameter, greater than 0.
    """

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], alpha: float, beta: float) -> None:
        super(TorchGamma, self).__init__(scope)

        # register alpha and gamma as torch parameters
        self.register_parameter("alpha", Parameter(torch.tensor(alpha, dtype=torch.float32)))
        self.register_parameter("beta", Parameter(torch.tensor(beta, dtype=torch.float32)))

        # create Torch distribution with specified parameters
        self.dist = D.Gamma(concentration=self.alpha, rate=self.beta)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1, dtype=data.dtype)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = self.dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, alpha: float, beta: float) -> None:
        self.alpha.data = torch.tensor(alpha)
        self.beta.data = torch.tensor(beta)

    def get_params(self) -> Tuple[float, float]:
        return self.alpha.data.cpu().numpy(), self.beta.data.cpu().numpy()  # type: ignore


@dispatch(P.Gamma)  # type: ignore[no-redef]
def toTorch(node: P.Gamma) -> TorchGamma:
    return TorchGamma(node.scope, *node.get_params())


@dispatch(TorchGamma)  # type: ignore[no-redef]
def toNodes(torch_node: TorchGamma) -> P.Gamma:
    return P.Gamma(torch_node.scope, *torch_node.get_params())
