'''
Created on March 20, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.splitting.Base import split_data_by_clusters, preproc
from spn.structure.StatisticalTypes import Type
from spn.structure.leaves.typedleaves.TypedLeaves import type_mixture_leaf_factory


def make_planes(N, dim):
    result = np.zeros((N, dim))
    for i in range(N):
        result[i, :] = np.random.uniform(-1, 1, dim)

    return result / np.sqrt(np.sum(result * result, axis=1))[:, None]


def above(planes, data):
    nD = data.shape[0]
    nP = planes.shape[0]
    centered = data - np.mean(data, axis=0)
    result = np.zeros((nD, nP))
    for i in range(nD):
        for j in range(nP):
            result[i, j] = np.sum(planes[j, :] * centered[i, :]) > 0
    return result


def get_split_cols_random_partition(ohe=False):
    def split_cols_random_partitions(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, None, ohe)
        clusters = above(make_planes(1, local_data.shape[1]), data)[:, 0]

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_random_partitions


def get_split_cols_binary_random_partition(threshold, beta_a=4, beta_b=5):
    """
    Randomly partitions the columns into two clusters with percentage threshold
    (otherwise does not split)
    The percentage of splitting is drawn from a Beta distribution with parameters (beta_a, beta_b)
    """
    def split_cols_binary_random_partitions(local_data, ds_context, scope):
        # data = preproc(local_data, ds_context, None, ohe)

        #
        # with a certain percentage it may fail, such that row partitioning may happen
        rand_gen = ds_context.rand_gen
        clusters = None
        p = rand_gen.rand()
        print('P', p)
        if p > threshold:
            #
            # draw percentage of split from  a Beta
            alloc_perc = rand_gen.beta(a=beta_a, b=beta_b)
            clusters = rand_gen.choice(2, size=local_data.shape[1], p=[alloc_perc,
                                                                       1 - alloc_perc])
            print(clusters, clusters.sum(), clusters.shape, alloc_perc)
        else:
            clusters = np.zeros(local_data.shape[1])

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_binary_random_partitions


def get_split_rows_binary_random_partition(beta_a=2, beta_b=5):
    """
    The percentage of splitting is drawn from a Beta distribution with parameters (beta_a, beta_b)

    """
    def split_rows_binary_random_partition(local_data, ds_context, scope):
        # data = preproc(local_data, ds_context, pre_proc, ohe)

        rand_gen = ds_context.rand_gen
        #
        # draw percentage of split from  a Beta
        alloc_perc = rand_gen.beta(a=beta_a, b=beta_b)
        clusters = rand_gen.choice(2, size=local_data.shape[0], p=[alloc_perc,
                                                                   1 - alloc_perc])

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_binary_random_partition


from copy import deepcopy
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.Sampling import sample_parametric_node
import scipy.stats
from spn.io.Text import to_JSON, spn_to_str_equation


def draw_params_gaussian_prior(nig_prior, rand_gen):
    sigma2_sam = scipy.stats.invgamma.rvs(a=nig_prior.a_0, size=1,
                                          # scale=1.0 / b_n,
                                          random_state=rand_gen)
    sigma2_sam = sigma2_sam * nig_prior.b_0
    std_n = np.sqrt(sigma2_sam * nig_prior.V_0)
    mu_sam = sample_parametric_node(Gaussian(nig_prior.m_0, std_n), 1, rand_gen)
    # print('sigm', sigma2_sam, 'std_n', std_n, 'v_n', V_n, mu_sam, m_n)

    #
    # updating params
    mean = mu_sam[0]
    # node.stdev = np.sqrt(node.variance)
    stdev = np.sqrt(sigma2_sam)[0]

    return {'mean': mean, 'stdev': stdev}


def draw_params_gamma_prior(gamma_prior, defaults, rand_gen):

    rate_sam = sample_parametric_node(Gamma(gamma_prior.a_0,
                                            gamma_prior.b_0), 1, rand_gen)

    #
    # updating params (only scale)
    beta = rate_sam[0]
    return {'alpha': defaults['alpha'],
            'beta': beta}


def draw_params_exponential_prior(gamma_prior, rand_gen):
    lambda_sam = sample_parametric_node(Gamma(gamma_prior.a_0,
                                              gamma_prior.b_0), 1, rand_gen)
    return {'l': lambda_sam}


def draw_params_categorical_prior(dir_prior,  defaults, rand_gen):
    alphas_0 = rand_gen.dirichlet(np.array([defaults['hyper-p']
                                            for j in range(defaults['k'])]), size=1)[0, :]
    p_sam = rand_gen.dirichlet(alphas_0, size=1)[0, :]
    return {'p': p_sam}


def draw_params_geometric_prior(beta_prior, rand_gen):
    p_sam = rand_gen.beta(a=beta_prior.a_0, b=beta_prior.b_0, size=1)
    return {'p': p_sam}


def draw_params_poisson_prior(gamma_prior, rand_gen):
    lambda_sam = sample_parametric_node(Gamma(gamma_prior.a_0,
                                              gamma_prior.b_0), 1, rand_gen)
    return {'mean': lambda_sam}


def draw_params_from_prior(param_class, prior, defaults, rand_gen):

    rand_params = None

    if param_class == Gaussian:
        rand_params = draw_params_gaussian_prior(prior, rand_gen)
    elif param_class == Gamma:
        rand_params = draw_params_gamma_prior(prior, defaults, rand_gen)
    elif param_class == Exponential:
        rand_params = draw_params_exponential_prior(prior, rand_gen)
    elif param_class == Categorical:
        rand_params = draw_params_categorical_prior(prior, defaults, rand_gen)
    elif param_class == Geometric:
        rand_params = draw_params_geometric_prior(prior, rand_gen)
    elif param_class == Poisson:
        rand_params = draw_params_poisson_prior(prior, rand_gen)
    else:
        raise ValueError('Unrecognized distribution to fit {}'.format(param_class))

    return rand_params


def random_params_from_priors(param_map, defaults, priors, rand_gen):
    """
    WRITEME
    """
    rand_type_param_map = deepcopy(param_map)
    for _type, param_types in param_map.items():
        for param_class, param_map in param_types.items():
            p_defaults = defaults.get(param_class)
            rand_param_map = draw_params_from_prior(param_class,
                                                    prior=priors[param_class],
                                                    defaults=p_defaults,
                                                    rand_gen=rand_gen)
            rand_type_param_map[_type][param_class]['params'] = rand_param_map

    return rand_type_param_map


MIN_K_CAT = 5
MAX_K_CAT = 15
MIN_ALPHA_GAMMA = 5
MAX_ALPHA_GAMMA = 25
MAX_HYPER_P_CAT = 10


PARAM_FORM_TYPE_MAP = {
    Gaussian: Type.REAL,
    Gamma: Type.POSITIVE,
    Exponential: Type.POSITIVE,
    LogNormal: Type.POSITIVE,
    Categorical: Type.CATEGORICAL,
    Geometric: Type.COUNT,
    Poisson: Type.COUNT,
    Bernoulli: Type.BINARY,
    NegativeBinomial: Type.COUNT,
    Hypergeometric: Type.COUNT
}


def create_random_unconstrained_type_mixture_leaf(data, ds_context, scope,
                                                  min_k=MIN_K_CAT,  max_k=MAX_K_CAT,
                                                  max_hyper_p_cat=MAX_HYPER_P_CAT,
                                                  min_alpha=MIN_ALPHA_GAMMA,
                                                  max_alpha=MAX_ALPHA_GAMMA):
    """
    Method to be employed by LearnSPN-like pipeline to create a type leaf, based on convext parameters
    """
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    idx = scope[0]
    rand_gen = ds_context.rand_gen
    meta_type = ds_context.meta_types[idx]
    true_type = ds_context.types[idx]
    param_map = ds_context.param_form_map[meta_type]
    priors = ds_context.priors

    allowed_param_forms = []
    for tm, t_map in param_map.items():
        for p_class, p_map in t_map.items():
            allowed_param_forms.append((p_class, p_map))
    # n_param_forms = int(np.sum([len(t_map) for tm, t_map in param_map.items()]))
    n_param_forms = len(allowed_param_forms)
    print(n_param_forms, 'meta type', meta_type, 'true type', true_type,
          'allowed forms', allowed_param_forms)

    #
    # random init weights: only 1.0 over the true type
    # rand_init_weights = np.zeros(n_param_forms)
    rand_init_weights = {}
    allowed_types = np.array([PARAM_FORM_TYPE_MAP[p_c] == true_type for
                              p_c, p_map in allowed_param_forms], dtype=bool)
    n_types = int(allowed_types.sum())
    print('Allowed types', allowed_types, n_types)
    inv_type_map = {}
    j = 0
    for i, t in enumerate(allowed_types):
        if t:
            inv_type_map[j] = i
            j += 1
    nonzero_weight_id = rand_gen.choice(n_types)
    nonzero_weight_id = inv_type_map[nonzero_weight_id]
    for j, (p_c, _p_map) in enumerate(allowed_param_forms):
        if j == nonzero_weight_id:
            rand_init_weights[p_c] = 1.0
        else:
            rand_init_weights[p_c] = 0.0
    print('Selected weights', rand_init_weights)
    assert np.array([v for v in rand_init_weights.values()]).sum() == 1.0

    #
    # random defaults
    defaults = {Categorical: {'k': rand_gen.choice(range(min_k, max_k)),
                              'hyper-p': rand_gen.choice(max_hyper_p_cat) + 1},
                Gamma: {'alpha': rand_gen.choice(range(min_alpha, max_alpha))}}
    print('\n\trandom default params for gamma and categorical:\n\t\t{}'.format(defaults))

    #
    # random parameters
    param_map = random_params_from_priors(param_map, defaults, priors, rand_gen)
    print('\n\trandom default params for gamma and categorical:\n\t\t{}'.format(defaults))

    leaf, _leaf_prior = type_mixture_leaf_factory(leaf_type='pm',
                                                  leaf_meta_type=meta_type,
                                                  type_to_param_map=param_map,
                                                  scope=scope,
                                                  init_weights=rand_init_weights)
    print('\nCreated random type leaf: {}'.format(spn_to_str_equation(leaf)))

    return leaf
