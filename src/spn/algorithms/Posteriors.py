'''
Created on April 10, 2018

@author: Antonio Vergari
@author: Alejandro Molina

'''
import numpy as np
import scipy

from spn.structure.leaves.parametric.Parametric import Gaussian, Gamma, LogNormal, Poisson, Categorical, Bernoulli, \
    Geometric, Exponential
from spn.structure.leaves.parametric.Sampling import sample_parametric_node


class PriorNormalInverseGamma:
    def __init__(self, m_0, V_0, a_0, b_0):
        self.m_0 = m_0
        self.V_0 = V_0
        self.a_0 = a_0
        self.b_0 = b_0


class PriorGamma:
    def __init__(self, a_0, b_0):
        self.a_0 = a_0
        self.b_0 = b_0


class PriorNormal:
    def __init__(self, mu_0, tau_0):
        self.mu_0 = mu_0
        self.tau_0 = tau_0


class PriorDirichlet:
    def __init__(self, alphas_0):
        self.alphas_0 = alphas_0


class PriorBeta:
    def __init__(self, a_0, b_0):
        self.a_0 = a_0
        self.b_0 = b_0


PARAM_PRIOR_MAP = {Gaussian: PriorNormalInverseGamma,
                   Gamma: PriorGamma,
                   LogNormal: PriorNormal,
                   Poisson: PriorGamma,
                   Categorical: PriorDirichlet,
                   Bernoulli: PriorBeta,
                   Geometric: PriorBeta,
                   Exponential: PriorGamma}


def update_parametric_parameters_posterior(node, X, rand_gen, prior):
    x = X[node.row_ids, node.scope]
    miss_vals = np.isnan(x)
    x = x[~miss_vals]

    if isinstance(node, Gaussian):
        update_params_GaussianNode(node, x, rand_gen, prior)
    elif isinstance(node, Gamma):
        update_params_GammaFixAlphaNode(node, x, rand_gen, prior)
    elif isinstance(node, LogNormal):
        update_params_LogNormalFixVarNode(node, x, rand_gen, prior)
    elif isinstance(node, Poisson):
        update_params_PoissonNode(node, x, rand_gen, prior)
    elif isinstance(node, Categorical):
        update_params_CategoricalNode(node, x, rand_gen, prior)
    elif isinstance(node, Bernoulli):
        update_params_BernoulliNode(node, x, rand_gen, prior)
    elif isinstance(node, Geometric):
        update_params_GeometricNode(node, x, rand_gen, prior)
    elif isinstance(node, Exponential):
        update_params_ExponentialNode(node, x, rand_gen, prior)
    else:
        raise Exception('Node type unknown: ' + str(type(node)))

    return


def update_params_GaussianNode2(node, X, rand_gen, nig_prior):
    """
    The prior over parameters is a Normal - Inverse - Gamma(NIG)


    [1] - Murphy K., Conjugate Bayesian analysis of the Gaussian distribution(2007)
          https: // www.cs.ubc.ca / ~murphyk / Papers / bayesGauss.pdf
          https://en.wikipedia.org/wiki/Conjugate_prior
          http://thaines.com/content/misc/gaussian_conjugate_prior_cheat_sheet.pdf
          ** http://homepages.math.uic.edu/~rgmartin/Teaching/Stat591/Bayes/Notes/591_gibbs.pdf
          ** https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf

    Return a sample for the node.params drawn from the posterior distribution
    which for conjugacy is still a NIG

    p(\mu, \sigma ^ 2, | X) = NIG(m_n, V_n, a_n, b_n)

    see[1]
    """

    assert isinstance(nig_prior, PriorNormalInverseGamma), nig_prior

    n = len(X)
    X_hat = np.mean(X)

    mean = (nig_prior.V_0 * nig_prior.m_0 + n * X_hat) / (nig_prior.V_0 + n)

    v = nig_prior.V_0 + n

    a = nig_prior.a_0 + n / 2

    b = nig_prior.b_0 + (n / 2) * (np.var(X) + (v / (v + n)) * np.power(X_hat - mean, 2))

    inv_sigma2_sam = sample_parametric_node(Gamma(a, b), 1, rand_gen)

    sigma2_sam = 1 / inv_sigma2_sam

    mu_sam = sample_parametric_node(Gaussian(mean, sigma2_sam / v), 1, rand_gen)

    # updating params
    node.mean = mu_sam[0]
    # node.stdev = np.sqrt(node.variance)
    node.stdev = np.sqrt(sigma2_sam)[0]


def update_params_GaussianNode(node, X, rand_gen, nig_prior):
    """
    The prior over parameters is a Normal - Inverse - Gamma(NIG)

    p(\mu, \sigma ^ 2) = NIG(m_0, V_0, a_0, b_0) =
                     = N(\mu | m_0, \sigma ^ {2}V_0)IG(\sigma ^ {2} | a_0, b_0)

    see[1], eq. 190 - 191

    [1] - Murphy K., Conjugate Bayesian analysis of the Gaussian distribution(2007)
          https: // www.cs.ubc.ca / ~murphyk / Papers / bayesGauss.pdf

    Return a sample for the node.params drawn from the posterior distribution
    which for conjugacy is still a NIG

    p(\mu, \sigma ^ 2, | X) = NIG(m_n, V_n, a_n, b_n)

    see[1]
    """

    assert isinstance(nig_prior, PriorNormalInverseGamma), nig_prior

    N = len(X)

    # N = len(node.row_ids)

    # eq (197)
    inv_V_0 = 1.0 / nig_prior.V_0
    inv_V_n = (inv_V_0 + N)
    V_n = 1 / inv_V_n

    # eq (198), just switching from avg to sum to prevent nans in numpy
    # when there are no instances assigned, it should be like sampling from the prior
    # x = X[node.row_ids, node.scope]
    # avg_x = x.mean()
    # m_n = (inv_V_0 * node.m_0 + N * avg_x) * V_n
    sum_x = X.sum()
    avg_x = sum_x / N if N else 0
    m_n = (inv_V_0 * nig_prior.m_0 + sum_x) * V_n

    # eq (199)
    # inv_V_n = 1.0 / V_n
    a_n = nig_prior.a_0 + N / 2
    # mu_n_hat = - m_n * m_n * inv_V_na
    # b_n = node.b_0 + (node.m_0 * node.m_0 * inv_V_0 +
    #                   np.dot(x, x) - m_n * m_n * inv_V_n
    #                   # (x * x - mu_n_hat).sum()
    #                   ) / 2
    b_n = nig_prior.b_0 + (np.dot(X - avg_x, X - avg_x) +
                           (N * inv_V_0 * (avg_x - nig_prior.m_0) * (avg_x - nig_prior.m_0)) * V_n) / 2

    #
    # sampling
    # first sample the variance from IG, then the mean from a N
    # see eq (191) and
    # TODO, optimize it with numba
    sigma2_sam = scipy.stats.invgamma.rvs(a=a_n, size=1,
                                          # scale=1.0 / b_n,
                                          random_state=rand_gen)
    sigma2_sam = sigma2_sam * b_n
    std_n = np.sqrt(sigma2_sam * V_n)
    mu_sam = sample_parametric_node(Gaussian(m_n, std_n), 1, None, rand_gen)
    # print('sigm', sigma2_sam, 'std_n', std_n, 'v_n', V_n, mu_sam, m_n)

    #
    # updating params
    node.mean = mu_sam[0]
    # node.stdev = np.sqrt(node.variance)
    node.stdev = np.sqrt(sigma2_sam)[0]


def update_params_GammaFixAlphaNode(node, X, rand_gen, gamma_prior):
    """
    The prior over \beta is again a Gamma distribution

    p(\beta) = Gamma(a_0, b_0)

    with shape \alpha_0 = a_0 and rate \beta_0 = b_0

    see[1], eq. (52 - 54), considering the inverse of the scale, the rate \frac{1}{\beta}
        and [2]

    [1] - Fink, D. A Compendium of Conjugate Priors(1997)
          https: // www.johndcook.com / CompendiumOfConjugatePriors.pdf
    [2] - https: // en.wikipedia.org / wiki / Conjugate_prior

    Return a sample for the node.params drawn from the posterior distribution
    which for conjugacy is still a Gamma

    p(\beta, | X) = Gamma(a_n, b_n)

    see[1, 2]
    """

    assert isinstance(gamma_prior, PriorGamma)

    N = len(X)

    #
    # if N is 0, then it would be like sampling from the prior
    # a_n = a_0 + N * alpha
    a_n = gamma_prior.a_0 + N * node.alpha
    # print(a_n, gamma_prior.a_0, N, node.alpha)

    #
    # x = X[node.row_ids, node.scope]
    sum_x = X.sum()
    b_n = gamma_prior.b_0 + sum_x

    #
    # sampling
    # TODO, optimize it with numba
    rate_sam = sample_parametric_node(Gamma(a_n, b_n), 1,  None, rand_gen)

    #
    # updating params (only scale)
    node.beta = rate_sam[0]


def update_params_LogNormalFixVarNode(node, X, rand_gen, normal_prior):
    """
    The prior over \mu is a Normal distribution

    p(\mu) = Normal(mu_0, tau_0)

    with mean mu_0 and precision(inverse variance) tau_0

    see[1]

    [1] - https: // en.wikipedia.org / wiki / Conjugate_prior

    Return a sample for the node.params drawn from the posterior distribution
    which for conjugacy is still a Normal

    p(\mu, | X) = Normal(mu_n, tau_n)

    see[1]
    """

    assert isinstance(normal_prior, PriorNormal)

    N = len(X)

    #
    # if N is 0, then it would be like sampling from the prior
    tau_n = normal_prior.tau_0 + N * node.precision

    #
    # x = X[node.row_ids, node.scope]
    log_sum_x = np.log(X).sum() if N > 0 else 0
    mu_n = (log_sum_x * node.precision + normal_prior.tau_0 * normal_prior.mu_0) / tau_n
    sum_x = X.sum()
    # mu_n = (sum_x * node.precision + node.tau_0 * node.mu_0) / tau_n

    #
    # sampling
    # TODO, optimize it with numba
    std_n = 1.0 / np.sqrt(tau_n)
    # print('STDN', std_n, tau_n, mu_n, log_sum_x)

    mu_sam = sample_parametric_node(Gaussian(mu_n, std_n), 1,  None, rand_gen)
    # print('STDN', std_n, tau_n, mu_n, sum_x, np.log(mu_sam), mu_sam)
    #
    # updating params (only mean)
    node.mean = mu_sam[0]


def update_params_PoissonNode(node, X, rand_gen, gamma_prior):
    """
    The prior over \lambda is a Gamma distribution

    p(\lambda) = Gamma(a_0, b_0)

    with shape \alpha_0 = a_0 and scale \beta_0 = b_0

    see[1]

    [1] - https: // en.wikipedia.org / wiki / Conjugate_prior

    Return a sample for the node.params drawn from the posterior distribution
    which for conjugacy is still a Gamma

    p(\lambda, | X) = Gamma(a_n, b_n)

    see[1]
    """

    assert isinstance(gamma_prior, PriorGamma)

    N = len(X)

    #
    # if N is 0, then it would be like sampling from the prior
    # x = X[node.row_ids, node.scope]
    sum_x = X.sum()
    a_n = gamma_prior.a_0 + sum_x
    b_n = gamma_prior.b_0 + N

    #
    # sampling
    # TODO, optimize it with numba
    lambda_sam = sample_parametric_node(Gamma(a_n, b_n), 1,  None, rand_gen)
    lambda_sam = lambda_sam  # / b_n

    #
    # updating params
    node.mean = lambda_sam[0]


def update_params_CategoricalNode(node, X, rand_gen, dir_prior):
    """
    The prior over parameters is a Dirichlet

    p(\{\pi_{k}\}) = Dir(\boldsymbol\alpha_0)

    see[1]

    [1] - https: // en.wikipedia.org / wiki / Conjugate_prior

    p(\{\pi_{k}\}|X) = Dir(\boldsymbol\alpha_n)

    see[1]
    """

    assert isinstance(dir_prior, PriorDirichlet)

    N = len(X)

    # x = X[node.row_ids, node.scope]
    n_counts = np.zeros(node.k)

    if N > 0:
        obs_vals, counts = np.unique(X, return_counts=True)
        obs_vals = obs_vals.astype(np.int64)
        n_counts[obs_vals] = counts

    alphas_n = dir_prior.alphas_0 + n_counts

    #
    # sampling
    p_sam = rand_gen.dirichlet(alphas_n, size=1)[0, :]

    #
    # updating params
    node.p = p_sam.tolist()


def update_params_BernoulliNode(node, X, rand_gen, beta_prior):
    """
    The prior over parameters is a Beta

    p(p) = Beta(\alpha_0=a_0, \beta_0=b_0)

    see[1]

    [1] - https: // en.wikipedia.org / wiki / Conjugate_prior

    p(p|X) = Beta(\alpha_n=a_n, \beta_n=b_n)

    see[1]
    """

    assert isinstance(beta_prior, PriorBeta)

    N = len(X)

    #
    # updating posterior parameters
    sum_x = X.sum()
    a_n = beta_prior.a_0 + sum_x
    b_n = beta_prior.b_0 + N - sum_x

    #
    # sampling
    p_sam = rand_gen.beta(a=a_n, b=b_n, size=1)

    #
    # updating params
    node.p = p_sam[0]


def update_params_GeometricNode(node, X, rand_gen, beta_prior):
    """
    The prior over parameters is a Beta

    p(p) = Beta(\alpha_0=a_0, \beta_0=b_0)

    see[1]

    [1] - https: // en.wikipedia.org / wiki / Conjugate_prior

    p(p|X) = Beta(\alpha_n=a_n, \beta_n=b_n)

    see[1]
    """

    assert isinstance(beta_prior, PriorBeta)

    N = len(X)

    #
    # updating posterior parameters
    sum_x = X.sum()
    a_n = beta_prior.a_0 + N
    b_n = beta_prior.b_0 + sum_x - N

    #
    # sampling
    p_sam = rand_gen.beta(a=a_n, b=b_n, size=1)

    #
    # updating params
    node.p = p_sam[0]


def update_params_ExponentialNode(node, X, rand_gen, gamma_prior):
    """
    The prior over the rate parameter is a Gamma

    p(\lambda) = Gamma(\alpha_0=a_0, \beta_0=b_0)

    see[1]

    [1] - https: // en.wikipedia.org / wiki / Conjugate_prior

    p(\lambda|X) = Gamma(\alpha_n=a_n, \beta_n=b_n)

    see[1]
    """

    assert isinstance(gamma_prior, PriorGamma)

    N = len(X)

    #
    # updating posterior parameters
    sum_x = X.sum()
    a_n = gamma_prior.a_0 + N
    b_n = gamma_prior.b_0 + sum_x

    #
    # sampling
    lambda_sam = sample_parametric_node(Gamma(a_n, b_n), 1,  None, rand_gen)
    lambda_sam = lambda_sam  # / b_n

    #
    # updating params
    node.l = lambda_sam[0]


###############################################################################
#
# POSTERIOR PREDICTIVE distributions
#


def posterior_predictive_GaussianNode(node, x_n_id, X, rand_gen, nig_prior):
    """
    The prior over parameters is a Normal - Inverse - Gamma(NIG)

    p(\mu, \sigma ^ 2) = NIG(m_0, V_0, a_0, b_0) =
                     = N(\mu | m_0, \sigma ^ {2}V_0)IG(\sigma ^ {2} | a_0, b_0)

    see[1], eq. 190 - 191

    The posterior predictive is (eq. 206)

    p(x_{n+1}|x_{1:n}) = T-Student_{2*a_n}(x_{new}|m_n, (b_n(1+V_n))/(a_n))

    [1] - Murphy K., Conjugate Bayesian analysis of the Gaussian distribution(2007)
          https: // www.cs.ubc.ca / ~murphyk / Papers / bayesGauss.pdf



    """

    x = X[node.row_ids, node.scope]
    miss_vals = np.isnan(x)
    if miss_vals.sum() > 0:
        raise ValueError('Posterior Predictive for Gaussian cannot deal with missing values')

    assert isinstance(nig_prior, PriorNormalInverseGamma)

    assert x_n_id not in node.row_ids

    x_n = X[x_n_id, node.scope]

    N = len(node.row_ids)

    # eq (197)
    inv_V_0 = 1.0 / nig_prior.V_0
    inv_V_n = (inv_V_0 + N)
    V_n = 1 / inv_V_n

    #
    # TODO: this is the same as in update the posterior parameters
    #

    # eq (198), just switching from avg to sum to prevent nans in numpy
    # when there are no instances assigned, it should be like sampling from the prior
    x = X[node.row_ids, node.scope]
    # avg_x = x.mean()
    # m_n = (inv_V_0 * node.m_0 + N * avg_x) * V_n
    sum_x = x.sum()
    avg_x = sum_x / N if N else 0
    m_n = (inv_V_0 * nig_prior.m_0 + sum_x) * V_n

    # eq (199)
    # inv_V_n = 1.0 / V_n
    a_n = nig_prior.a_0 + N / 2
    # mu_n_hat = - m_n * m_n * inv_V_na
    # b_n = node.b_0 + (node.m_0 * node.m_0 * inv_V_0 +
    #                   np.dot(x, x) - m_n * m_n * inv_V_n
    #                   # (x * x - mu_n_hat).sum()
    #                   ) / 2
    b_n = nig_prior.b_0 + (np.dot(x - avg_x, x - avg_x) +
                           (N * inv_V_0 * (avg_x - nig_prior.m_0) * (avg_x - nig_prior.m_0)) * V_n) / 2
    sigma_2n = b_n * (1 + V_n) / a_n

    return scipy.stats.t(df=2 * a_n, loc=m_n, scale=np.sqrt(sigma_2n)).pdf(x_n)
