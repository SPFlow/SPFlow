'''
Created on March 20, 2018
@author: Alejandro Molina
'''

import numpy as np

from spn.structure.Base import Leaf, Sum, get_nodes_by_type
from spn.structure.StatisticalTypes import Type


class Parametric(Leaf):
    def __init__(self, type, scope=None):
        Leaf.__init__(self, scope=scope)
        self._type = type

    @property
    def type(self):
        return self._type

    @property
    def params(self):
        raise Exception("Not Implemented")


class Gaussian(Parametric):
    """
    Implements a univariate gaussian distribution with parameters
    \mu(mean)
    \sigma ^ 2 (variance)
    (alternatively \sigma is the standard deviation(stdev) and \sigma ^ {-2} the precision)
    """

    def __init__(self, mean, stdev, scope=None):
        Parametric.__init__(self, Type.REAL, scope=scope)

        # parameters
        self.mean = mean
        self.stdev = stdev

    @property
    def params(self):
        return {'mean': self.mean, 'stdev': self.stdev}

    @property
    def precision(self):
        return 1.0 / self.variance

    @property
    def variance(self):
        return self.stdev * self.stdev

    @property
    def mode(self):
        return self.mean


class Uniform(Parametric):
    def __init__(self, density, type, scope=None):
        Parametric.__init__(self, type, scope=scope)

        # parameters
        self.density = density

    @property
    def params(self):
        return {'density': self.density}


class Gamma(Parametric):
    """
    Implements a univariate Gamma distribution with parameter
    \beta(scale)

    where \alpha(shape) is known and fixed

    """

    def __init__(self, alpha, beta, scope=None):
        Parametric.__init__(self, Type.POSITIVE, scope=scope)

        # parameters
        self.alpha = alpha
        self.beta = beta

    @property
    def params(self):
        return {'alpha': self.alpha, 'beta': self.beta}

    @property
    def mode(self):
        return (self.alpha - 1) / self.beta


class LogNormal(Parametric):
    """
    Implements a univariate Log - Normal distribution with parameter
    \mu(shape, mean)

    where the precition \tau(shape) is known and fixed.
    """

    def __init__(self, mean, stdev, scope=None):
        Parametric.__init__(self, Type.POSITIVE, scope=scope)

        # parameters
        self.mean = mean
        self.stdev = stdev

    @property
    def params(self):
        return {'mean': self.mean, 'stdev': self.stdev}

    @property
    def variance(self):
        return self.stdev * self.stdev

    @property
    def precision(self):
        return 1 / self.variance

    @property
    def mode(self):
        return np.exp(self.mean - self.variance)
        # return np.exp(self.mean)


class Poisson(Parametric):
    """
    Implements a univariate Poisson distribution with parameter
    \lambda (mean)
    """

    def __init__(self, mean, scope=None):
        Parametric.__init__(self, Type.COUNT, scope=scope)

        self.mean = mean

    @property
    def params(self):
        return {'mean': self.mean}

    @property
    def mode(self):
        return np.floor(self.mean)


class Bernoulli(Parametric):
    """
    Implements a univariate Bernoulli distribution with parameter
    p (probability of a success)
    """

    def __init__(self, p, scope=None):
        Parametric.__init__(self, Type.BINARY, scope=scope)

        self.p = p

    @property
    def params(self):
        return {'p': self.p}

    @property
    def mode(self):
        if self.p > 0:
            return 1
        else:
            return 0


class NegativeBinomial(Parametric):
    """
    Implements a univariate NegativeBinomial distribution with  parameter
    p (probability of a success) and number of non-random number of successes occurs.

    FIXME: mismatch from wiki to scipy
    """

    def __init__(self, n, p, scope=None):
        Parametric.__init__(self, Type.COUNT, scope=scope)

        self.n = n
        self.p = p

    @property
    def params(self):
        return {'p': self.p, 'n': self.n}

    @property
    def mode(self):
        if self.n <= 1:
            return 0
        else:
            return np.floor(self.p * (self.n - 1) / (1 - self.p))


class Hypergeometric(Parametric):
    """
    Implements a univariate Hypergeometric distribution with  parameter
    of k successes in n trials, without replacement, from a finite population of size N that contains exactly K objects with that feature

    FIXME: mismatch in the wiki in the conjugate prior table
    """

    def __init__(self, K, N, n, scope=None):
        Parametric.__init__(self, Type.COUNT, scope=scope)

        self.n = n
        self.K = K
        self.N = N

    @property
    def params(self):
        return {'N': self.N, 'K': self.K, 'n': self.n}

    @property
    def mode(self):
        return np.floor((self.n + 1) * (self.K + 1 / (self.N + 2)))


class Geometric(Parametric):
    """
    Implements a univariate Geometric distribution with  parameter
    p,  the probability of success on each trial

    """

    def __init__(self, p, scope=None):
        Parametric.__init__(self, Type.COUNT, scope=scope)

        self.p = p

    @property
    def params(self):
        return {'p': self.p}

    @property
    def mode(self):
        # return 0  # or 1? check wiki
        return 1


class Categorical(Parametric):
    """
    Implements a univariate categorical distribution with $k$ parameters
    {\pi_{k}}

    representing the probability of the k-th category

    The conjugate prior for these values would be a Dirichlet

    p(\{\pi_{k}\}) = Dir(\boldsymbol\alpha)
    """

    def __init__(self, p, scope=None):
        Parametric.__init__(self, Type.CATEGORICAL, scope=scope)

        # parameters
        assert np.isclose(np.sum(p), 1), 'Probabilities p shall sum to 1'
        self.p = p

    @property
    def params(self):
        return {"p": self.p}

    @property
    def k(self):
        return len(self.p)

    @property
    def mode(self):
        return np.argmax(self.p)

    def sample(self, n_samples, rand_gen):
        return rand_gen.choice(np.arange(self.k), p=self._p, size=n_samples)


class Exponential(Parametric):
    """
    Implements a univariate Exponential distribution with  parameter
    \lambda,  the rate of the distribution

    """

    def __init__(self, l, scope=None):
        Parametric.__init__(self, Type.POSITIVE, scope=scope)

        self.l = l

    @property
    def params(self):
        return {'l': self.l}

    @property
    def mode(self):
        return 0
