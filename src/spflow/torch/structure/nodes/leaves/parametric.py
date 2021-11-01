"""
Created on July 4, 2021
@authors: Philipp Deibert
"""

from abc import ABC
from typing import List, Union, Tuple, Any, Optional

from multipledispatch import dispatch  # type: ignore

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.torch.structure.nodes.node import TorchLeafNode
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType
import spflow.base.structure.nodes.leaves.parametric.parametric as P

from scipy.special import comb  # type: ignore

def proj_real_to_bounded(x: torch.Tensor, lb: Optional[Union[float, torch.Tensor]]=None, ub: Optional[Union[float, torch.Tensor]]=None) -> torch.Tensor:
    """Projects the real numbers onto a bounded interval.
    """
    if(lb is not None and ub is not None):
        # project to bounded interval
        return torch.sigmoid(x)*(ub-lb) + lb
    elif(ub is None):
        # project to left-bounded interval
        return torch.exp(x) + lb
    else:
        # project to right-bounded interval
        return -torch.exp(x) + ub

def proj_bounded_to_real(x: torch.Tensor, lb: Optional[Union[float, torch.Tensor]]=None, ub: Optional[Union[float, torch.Tensor]]=None) -> torch.Tensor:
    """Projects a bounded interval onto the real numbers.
    """
    if(lb is not None and ub is not None):
        # project from bounded interval
        return torch.log( (x-lb)/(ub-x) )
    elif(ub is None):
        # project from left-bounded interval
        return torch.log(x-lb)
    else:
        # project from right-bounded interval
        return torch.log(ub-x)

def proj_positive_definite_to_triangular(A: torch.Tensor) -> torch.Tensor:
    L: torch.Tensor = torch.cholesky(A)
    return L

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
        mean:
            mean (mu) of the distribution.
        stdev:
            standard deviation (sigma) of the distribution.
    """

    ptype = ParametricType.CONTINUOUS

    def __init__(self, scope: List[int], mean: float, stdev: float) -> None:
        super(TorchGaussian, self).__init__(scope)

        if len(scope) != 1:
            raise ValueError("Invalid scope size for univariate Gaussian")

        # register mean as torch parameter
        self.register_parameter("mean", Parameter())
        # register auxiliary torch paramter for standard deviation 
        self.register_parameter("stdev_aux", Parameter())

        # set parameters
        self.set_params(mean, stdev)

    def __getattr__(self, attr: str) -> Any:
        if(attr == "stdev"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.stdev_aux, lb=0.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.Normal(loc=self.mean, scale=self.stdev)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, mean: float, stdev: float) -> None:

        if not (np.isfinite(mean) and np.isfinite(stdev)):
            raise ValueError(
                f"Mean and standard deviation for Gaussian distribution must be finite, but were: {mean}, {stdev}"
            )
        if stdev <= 0.0:
            raise ValueError(
                f"Standard deviation for Gaussian distribution must be greater than 0.0, but was: {stdev}"
            )

        self.mean.data = torch.tensor(float(mean))
        self.stdev_aux.data = proj_bounded_to_real(torch.tensor(float(stdev)), lb=0.0)

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
    PDF(x) =
        1/(x*sigma*sqrt(2*pi) * exp(-(ln(x)-mu)^2/(2*sigma^2)), where
            - x is an observation
            - mu is the mean
            - sigma is the standard deviation
    Attributes:
        mean:
            mean (mu) of the distribution.
        stdev:
            standard deviation (sigma) of the distribution (must be greater than 0).
    """

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], mean: float, stdev: float) -> None:
        super(TorchLogNormal, self).__init__(scope)

        # register mean as torch parameter
        self.register_parameter("mean", Parameter())
        # register auxiliary torch paramter for standard deviation 
        self.register_parameter("stdev_aux", Parameter())

        # set parameters
        self.set_params(mean, stdev)
    
    def __getattr__(self, attr: str) -> Any:
        if(attr == "stdev"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.stdev_aux, lb=0.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.LogNormal(loc=self.mean, scale=self.stdev)

        # test data for distribution support
        support_mask = dist.support.check(scope_data).sum(dim=1) == scope_data.shape[1]

        # set probability of data outside of support to -inf
        log_prob[~support_mask] = -float("Inf")

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask & support_mask] = dist.log_prob(
            scope_data[prob_mask & support_mask]
        )

        return log_prob

    def set_params(self, mean: float, stdev: float) -> None:

        if not (np.isfinite(mean) and np.isfinite(stdev)):
            raise ValueError(
                f"Mean and standard deviation for Gaussian distribution must be finite, but were: {mean}, {stdev}"
            )
        if stdev <= 0.0:
            raise ValueError(
                f"Standard deviation for Gaussian distribution must be greater than 0.0, but was: {stdev}"
            )

        self.mean.data = torch.tensor(float(mean))
        self.stdev_aux.data = proj_bounded_to_real(torch.tensor(float(stdev)), lb=0.0)

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
        1/sqrt((2*pi)^d * det(cov)) * exp(-1/2 (x-mu)^T * cov^(-1) * (x-mu)), where
            - d is the dimension of the distribution
            - x is the d-dim. vector of observations
            - mu is the d-dim. mean_vector
            - cov is the dxd covariance_matrix
    Attributes:
        mean_vector:
            A list, NumPy array or a PyTorch tensor holding the means (mu) of each of the one-dimensional Normal distributions.
            Has exactly as many elements as the scope of this leaf.
        covariance_matrix:
            A list of lists, NumPy array or PyTorch tensor (representing a two-dimensional NxN matrix, where N is the length
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

        if isinstance(mean_vector, list) or isinstance(mean_vector, np.ndarray):
            # convert float list or numpy array to torch tensor
            mean_vector = torch.Tensor(mean_vector)

        if isinstance(covariance_matrix, list) or isinstance(covariance_matrix, np.ndarray):
            # convert float list or numpy array to torch tensor
            covariance_matrix = torch.Tensor(covariance_matrix)

        # dimensions
        self.d = mean_vector.numel()

        # register mean vector as torch parameters
        self.register_parameter("mean_vector", Parameter())

        # internally we use the lower triangular matrix (Cholesky decomposition) to encode the covariance matrix
        # register (auxiliary) values for diagonal and non-diagonal values of lower triangular matrix as torch parameters
        self.register_parameter("tril_diag_aux", Parameter())
        self.register_parameter("tril_nondiag", Parameter())

        # pre-compute and store indices of non-diagonal values for lower triangular matrix
        self.tril_nondiag_indices = torch.tril_indices(self.d, self.d, offset=-1)

        #set parameters
        self.set_params(mean_vector, covariance_matrix)

    def __getattr__(self, attr: str) -> Any:
        if(attr == "covariance_tril"):
            # create zero matrix of appropriate dimension
            L_nondiag = torch.zeros(self.d, self.d)
            # fill non-diagonal values of lower triangular matrix
            L_nondiag[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]] = self.tril_nondiag
            # add (projected) diagonal values
            L = L_nondiag + proj_real_to_bounded(self.tril_diag_aux, lb=0.0)*torch.eye(self.d)
            # return lower triangular matrix
            return L
        elif(attr == "covariance_matrix"):
            # get lower triangular matrix
            L = self.covariance_tril
            # return covariance matrix
            return torch.matmul(L, L.T)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.MultivariateNormal(loc=self.mean_vector, scale_tril=self.covariance_tril)
        
        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = dist.log_prob(scope_data[prob_mask]).unsqueeze(-1)

        return log_prob

    def set_params(
        self,
        mean_vector: Union[List[float], torch.Tensor, np.ndarray],
        covariance_matrix: Union[List[List[float]], torch.Tensor, np.ndarray],
    ) -> None:

        if isinstance(mean_vector, list):
            # convert float list to torch tensor
            mean_vector = torch.tensor([float(v) for v in mean_vector])
        elif isinstance(mean_vector, np.ndarray):
            # convert numpy array to torch tensor
            mean_vector = torch.from_numpy(mean_vector).type(torch.get_default_dtype())

        if isinstance(covariance_matrix, list):
            # convert numpy array to torch tensor
            covariance_matrix = torch.tensor([[float(v) for v in row] for row in covariance_matrix])
        elif isinstance(covariance_matrix, np.ndarray):
            # convert numpy array to torch tensor
            covariance_matrix = torch.from_numpy(covariance_matrix).type(torch.get_default_dtype())

        # TODO: dimensions should be same size as scope

        # set mean vector
        self.mean_vector.data = mean_vector
        
        # compute lower triangular matrix
        L = torch.linalg.cholesky(covariance_matrix)

        # set diagonal and non-diagonal values of lower triangular matrix
        self.tril_diag_aux.data = proj_bounded_to_real(torch.diag(L), lb=0.0)
        self.tril_nondiag.data = L[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]]

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
        start:
            Start of the interval.
        end:
            End of interval (must be larger the interval start).
    """

    ptype = ParametricType.CONTINUOUS

    def __init__(self, scope: List[int], start: float, end: float) -> None:
        super(TorchUniform, self).__init__(scope)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))

        # set parameters
        self.set_params(start, end)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = ((scope_data >= self.start) & (scope_data <= self.end)).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = self.dist.log_prob(
            scope_data[prob_mask & support_mask]
        )

        return log_prob

    def set_params(self, start: float, end: float) -> None:

        if not start < end:
            raise ValueError(
                f"Lower bound for Uniform distribution must be less than upper bound, but were: {start}, {end}"
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"Lower and upper bound must be finite, but were: {start}, {end}")

        # since torch Uniform distribution excludes the upper bound, compute next largest number
        end_next = torch.nextafter(torch.tensor(end), torch.tensor(float("Inf")))

        self.start.data = torch.tensor(float(start))  # type: ignore
        self.end.data = torch.tensor(float(end_next))  # type: ignore

        # create Torch distribution with specified parameters
        self.dist = D.Uniform(low=self.start, high=end_next)

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
        p:
            Probability of success in the range [0,1].
    """

    ptype = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:
        super(TorchBernoulli, self).__init__(scope)

        # register auxiliary torch paramter for the success probability p
        self.register_parameter("p_aux", Parameter())

        # set parameters
        self.set_params(p)

    def __getattr__(self, attr: str) -> Any:
        if(attr == "p"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.Bernoulli(probs=self.p)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = ((scope_data == 1) | (scope_data == 0)).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = dist.log_prob(
            scope_data[prob_mask & support_mask].type(torch.get_default_dtype())
        )

        return log_prob

    def set_params(self, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Bernoulli distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)

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
        n:
            Number of i.i.d. Bernoulli trials (greater of equal to 0).
        p:
            Probability of success of each trial in the range [0,1].
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: float) -> None:
        super(TorchBinomial, self).__init__(scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # register auxiliary torch parameter for the success probability p
        self.register_parameter("p_aux", Parameter())

        # set parameters
        self.set_params(n, p)
    
    def __getattr__(self, attr: str) -> Any:
        if(attr == "p"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.Binomial(total_count=self.n, probs=self.p)

        # test data for distribution support
        support_mask = dist.support.check(scope_data)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = ((scope_data >= 0) & (scope_data <= self.n)).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = dist.log_prob(
            scope_data[prob_mask & support_mask]
        )

        return log_prob

    def set_params(self, n: int, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Binomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Binomial distribution must to greater of equal to 0, but was: {n}"
            )

        self.n.data = torch.tensor(int(n))  # type: ignore
        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)  # type: ignore

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
        n:
            Number of i.i.d. trials (greater of equal to 0).
        p:
            Probability of success of each trial in the range (0,1].
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: float) -> None:
        super(TorchNegativeBinomial, self).__init__(scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # register auxiliary torch parameter for the success probability p
        self.register_parameter("p_aux", Parameter())

        # set parameters
        self.set_params(n, p)
    
    def __getattr__(self, attr: str) -> Any:
        if(attr == "p"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        # note: the distribution is not stored as an attribute due to mismatching parameters after gradient updates (gradients don't flow back to p when initializing with 1.0-p)
        dist = D.NegativeBinomial(total_count=self.n, probs=torch.ones(1) - self.p)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = (scope_data >= 0).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = dist.log_prob(scope_data[prob_mask & support_mask])

        return log_prob

    def set_params(self, n: int, p: float) -> None:

        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for NegativeBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for NegativeBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        self.n.data = torch.tensor(int(n))  # type: ignore
        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)  # type: ignore

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
        l:
            Expected value (& variance) of the Poisson distribution (usually denoted as lambda).
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], l: float) -> None:
        super(TorchPoisson, self).__init__(scope)

        # register auxiliary torch parameter for lambda l
        self.register_parameter("l_aux", Parameter())

        # set parameters
        self.set_params(l)
    
    def __getattr__(self, attr: str) -> Any:
        if(attr == "l"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.l_aux, lb=0.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.Poisson(rate=self.l)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = (scope_data >= 0).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = dist.log_prob(
            scope_data[prob_mask & support_mask]
        )

        return log_prob

    def set_params(self, l: float) -> None:

        if not np.isfinite(l):
            raise ValueError(f"Value of l for Poisson distribution must be finite, but was: {l}")

        self.l_aux.data = proj_bounded_to_real(torch.tensor(float(l)), lb=0.0)

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
        p:
            Probability of success in the range (0,1].
    """

    ptype = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:
        super(TorchGeometric, self).__init__(scope)

        # register auxiliary torch parameter for the success probability p
        self.register_parameter("p_aux", Parameter())

        # set parameters
        self.set_params(p)
    
    def __getattr__(self, attr: str) -> Any:
        if(attr == "p"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.Geometric(probs=self.p)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = ((scope_data >= 1) & dist.support.check(scope_data)).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = dist.log_prob(
            scope_data[prob_mask & support_mask] - 1
        )

        return log_prob

    def set_params(self, p: float) -> None:

        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Geometric distribution must to be greater than 0.0 and less or equal to 1.0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)

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
        N:
            Total number of entities (in the population), greater or equal to 0.
        M:
            Number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
        n:
            Number of observed entities (sample size), greater or equal to zero and less than or equal to N.
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], N: int, M: int, n: int) -> None:
        super(TorchHypergeometric, self).__init__(scope)

        # register parameters as torch buffers (should not be changed)
        self.register_buffer("N", torch.empty(size=[]))
        self.register_buffer("M", torch.empty(size=[]))
        self.register_buffer("n", torch.empty(size=[]))

        # set parameters
        self.set_params(N, M, n)

    def log_prob(self, k: torch.Tensor):

        N_minus_M = self.N - self.M
        n_minus_k = self.n - k

        # ----- (M over m) * (N-M over n-k) / (N over n) -----

        # log_M_over_k = torch.lgamma(self.M+1) - torch.lgamma(self.M-k+1) - torch.lgamma(k+1)
        # log_NM_over_nk = torch.lgamma(N_minus_M+1) - torch.lgamma(N_minus_M-n_minus_k+1) - torch.lgamma(n_minus_k+1)
        # log_N_over_n = torch.lgamma(self.N+1) - torch.lgamma(self.N-self.n+1) - torch.lgamma(self.n+1)
        # result = log_M_over_k + log_NM_over_nk - log_N_over_n

        # ---- alternatively (more precise according to SciPy) -----
        # betaln(good+1, 1) + betaln(bad+1,1) + betaln(total-draws+1, draws+1) - betaln(k+1, good-k+1) - betaln(draws-k+1, bad-draws+k+1) - betaln(total+1, 1)

        # TODO: avoid recomputation of terms
        result = (
            torch.lgamma(self.M + 1)  # type: ignore
            + torch.lgamma(torch.tensor(1.0))
            - torch.lgamma(self.M + 2)  # type: ignore
            + torch.lgamma(N_minus_M + 1)  # type: ignore
            + torch.lgamma(torch.tensor(1.0))
            - torch.lgamma(N_minus_M + 2)  # type: ignore
            + torch.lgamma(self.N - self.n + 1)  # type: ignore
            + torch.lgamma(self.n + 1)  # type: ignore
            - torch.lgamma(self.N + 2)  # type: ignore
            - torch.lgamma(k + 1)
            - torch.lgamma(self.M - k + 1)
            + torch.lgamma(self.M + 2)  # type: ignore
            - torch.lgamma(n_minus_k + 1)
            - torch.lgamma(N_minus_M - self.n + k + 1)
            + torch.lgamma(N_minus_M + 2)  # type: ignore
            - torch.lgamma(self.N + 1)  # type: ignore
            - torch.lgamma(torch.tensor(1.0))
            + torch.lgamma(self.N + 2)  # type: ignore
        )

        return result

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = (
            ((scope_data >= max(0, self.n + self.M - self.N)) & (scope_data <= min(self.n, self.M)))
            .sum(dim=1)
            .bool()
        )
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = self.log_prob(scope_data[prob_mask & support_mask])

        return log_prob

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

        self.M.data = torch.tensor(int(M))
        self.N.data = torch.tensor(int(N))
        self.n.data = torch.tensor(int(n))

    def get_params(self) -> Tuple[int, int, int]:
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
        l:
            Rate parameter of the Exponential distribution (usually denoted as lambda, must be greater than 0).
    """

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], l: float) -> None:
        super(TorchExponential, self).__init__(scope)

        # register auxiliary torch parameter for parameter l
        self.register_parameter("l_aux", Parameter())

        # set parameters
        self.set_params(l)
    
    def __getattr__(self, attr: str) -> Any:
        if(attr == "l"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.l_aux, lb=0.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.Exponential(rate=self.l)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = (scope_data >= 0).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = dist.log_prob(
            scope_data[prob_mask & support_mask]
        )

        return log_prob

    def set_params(self, l: float) -> None:

        if l <= 0.0 or not np.isfinite(l):
            raise ValueError(
                f"Value of l for Exponential distribution must be greater than 0, but was: {l}"
            )

        self.l_aux.data = proj_bounded_to_real(torch.tensor(float(l)), lb=0.0)

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
        alpha:
            Shape parameter, greater than 0.
        beta:
            Scale parameter, greater than 0.
    """

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], alpha: float, beta: float) -> None:
        super(TorchGamma, self).__init__(scope)

        # register auxiliary torch parameters for alpha and beta
        self.register_parameter("alpha_aux", Parameter())
        self.register_parameter("beta_aux", Parameter())

        # set parameters
        self.set_params(alpha, beta)
    
    def __getattr__(self, attr: str) -> Any:
        if(attr == "alpha"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.alpha_aux, lb=0.0)
        if(attr == "beta"):
            # project auxiliary parameter onto actual parameter range
            return proj_real_to_bounded(self.beta_aux, lb=0.0)
        else:
            return nn.Module.__getattr__(self, attr)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.Gamma(concentration=self.alpha, rate=self.beta)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = dist.log_prob(scope_data[prob_mask])

        return log_prob

    def set_params(self, alpha: float, beta: float) -> None:

        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(
                f"Value of alpha for Gamma distribution must be greater than 0, but was: {alpha}"
            )
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError(
                f"Value of beta for Gamma distribution must be greater than 0, but was: {beta}"
            )

        self.alpha_aux.data = proj_bounded_to_real(torch.tensor(float(alpha)), lb=0.0)
        self.beta_aux.data = proj_bounded_to_real(torch.tensor(float(beta)), lb=0.0)

    def get_params(self) -> Tuple[float, float]:
        return self.alpha.data.cpu().numpy(), self.beta.data.cpu().numpy()  # type: ignore


@dispatch(P.Gamma)  # type: ignore[no-redef]
def toTorch(node: P.Gamma) -> TorchGamma:
    return TorchGamma(node.scope, *node.get_params())


@dispatch(TorchGamma)  # type: ignore[no-redef]
def toNodes(torch_node: TorchGamma) -> P.Gamma:
    return P.Gamma(torch_node.scope, *torch_node.get_params())
