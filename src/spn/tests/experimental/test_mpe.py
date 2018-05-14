import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import datetime
import os
import sys
import logging
import pickle
import gzip
import json
from collections import OrderedDict

import numpy as np
import numba

from spn.algorithms.Inference import likelihood, mpe, mpe_likelihood
from spn.algorithms.LearningWrappers import learn_rand_spn
from spn.structure.StatisticalTypes import MetaType, Type, META_TYPE_MAP
from spn.structure.Base import Context, Product, Sum, Node
from spn.structure.Base import Leaf, get_nodes_by_type, get_parent_map, assign_ids, rebuild_scopes_bottom_up
from spn.algorithms.Posteriors import update_parametric_parameters_posterior, PriorDirichlet, PriorGamma, \
    PriorNormal, PriorNormalInverseGamma, PriorBeta
from spn.structure.leaves.parametric.Parametric import Gaussian, Gamma, LogNormal, Categorical, Poisson, Parametric, Bernoulli, Geometric, Hypergeometric, NegativeBinomial, Exponential, type_mixture_leaf_factory, LEAF_TYPES, TypeLeaf, TypeMixture, TypeMixtureUnconstrained
from spn.structure.leaves.parametric.Parametric import get_type_partitioning_leaves, INV_TYPE_PARAM_MAP
from spn.structure.leaves.parametric.Text import add_parametric_text_support
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import set_leaf_params, set_omegas_params
from spn.io.Text import to_JSON, spn_to_str_equation

from spn.algorithms.Sampling import sample_instances
from spn.algorithms.Statistics import get_structure_stats_dict
from bin.spstd_model_ha1 import sp_infer_data_types_ha1, retrieve_best_sample

from visualize import visualize_data_partition, reorder_data_partitions, visualize_histogram, approximate_density, plot_distributions_fitting_data

import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    show_plots = False

    add_parametric_inference_support()
    add_parametric_text_support()

    x_range = np.array([-10, 10])
    #
    # testing MPE inference for the univ distributions

    #
    # gaussian
    gaussian = Gaussian(mean=0.5, stdev=2, scope=[0])

    pdf_x, pdf_y = approximate_density(gaussian, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="gaussian")
    plt.axvline(x=gaussian.mode, color='r')
    if show_plots:
        plt.show()

    #
    # gamma, alpha=1, beta=5
    gamma = Gamma(alpha=1, beta=5, scope=[0])

    pdf_x, pdf_y = approximate_density(gamma, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="gamma")
    plt.axvline(x=gamma.mode, color='r')
    if show_plots:
        plt.show()

    #
    # gamma, alpha=20, beta=5
    gamma = Gamma(alpha=20, beta=5, scope=[0])

    pdf_x, pdf_y = approximate_density(gamma, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="gamma")
    plt.axvline(x=gamma.mode, color='r')
    if show_plots:
        plt.show()

    #
    # gamma, alpha=20, beta=0.5
    gamma = Gamma(alpha=20, beta=0.5, scope=[0])

    pdf_x, pdf_y = approximate_density(gamma, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="gamma")
    plt.axvline(x=gamma.mode, color='r')
    if show_plots:
        plt.show()

    #
    # gamma, alpha=20, beta=1
    gamma = Gamma(alpha=20, beta=1.1, scope=[0])

    pdf_x, pdf_y = approximate_density(gamma, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="gamma")
    print('Gamma Mode:', gamma.mode)
    plt.axvline(x=gamma.mode, color='r')
    if show_plots:
        plt.show()

    #
    # lognormal
    lognormal = LogNormal(mean=-5, stdev=1.1, scope=[0])

    pdf_x, pdf_y = approximate_density(lognormal, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="lognormal")
    print('LogNormal Mode:', lognormal.mode)
    plt.axvline(x=lognormal.mode, color='r')
    if show_plots:
        plt.show()

    #
    # poisson
    poisson = Poisson(mean=5, scope=[0])

    pdf_x, pdf_y = approximate_density(poisson, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="poisson")
    print('Poisson Mode:', poisson.mode)
    plt.axvline(x=poisson.mode, color='r')
    if show_plots:
        plt.show()

    #
    # bernoulli
    bernoulli = Bernoulli(p=.7, scope=[0])

    pdf_x, pdf_y = approximate_density(bernoulli, [0.0, 1.0])
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="bernoulli")
    print('Bernoulli Mode:', bernoulli.mode)
    plt.axvline(x=bernoulli.mode, color='r')
    if show_plots:
        plt.show()

    #
    # NegativeBinomial
    # negativebinomial = NegativeBinomial(n=5, p=0.7, scope=[0])

    # pdf_x, pdf_y = approximate_density(negativebinomial, x_range)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(pdf_x, pdf_y, label="NegativeBinomial")
    # print('NegativeBinomial Mode:', negativebinomial.mode)
    # plt.axvline(x=negativebinomial.mode, color='r')
    # if show_plots:
    #     plt.show()

    #
    # geometric
    geometric = Geometric(p=.025, scope=[0])

    pdf_x, pdf_y = approximate_density(geometric, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="geometric")
    print('Geometric Mode:', geometric.mode)
    plt.axvline(x=geometric.mode, color='r')
    if show_plots:
        plt.show()

    #
    # categorical
    categorical = Categorical(p=[0.1, 0.05, 0.3, 0.05, 0.2, 0.2, 0.1], scope=[0])

    pdf_x, pdf_y = approximate_density(categorical, np.arange(categorical.k))
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="categorical")
    print('Categorical Mode:', categorical.mode)
    plt.axvline(x=categorical.mode, color='r')
    if show_plots:
        plt.show()

    #
    # exponential
    exponential = Exponential(l=5, scope=[0])

    pdf_x, pdf_y = approximate_density(exponential, x_range)
    fig, ax = plt.subplots(1, 1)
    ax.plot(pdf_x, pdf_y, label="exponential")
    print('Exponential Mode:', exponential.mode)
    plt.axvline(x=exponential.mode, color='r')
    if show_plots:
        plt.show()

    #
    #
    # testing for a product node over two leaves
    rand_gen = np.random.RandomState(17)

    l_1 = Gaussian(mean=0.5, stdev=2, scope=[0])
    l_2 = Poisson(mean=5, scope=[1])

    p = Product()
    p.children = [l_1, l_2]
    rebuild_scopes_bottom_up(p)
    assign_ids(p)

    N = 10
    X_1 = rand_gen.normal(loc=0, scale=1, size=(N, 1))
    X_2 = rand_gen.poisson(lam=3, size=(N, 1))
    X = np.concatenate((X_1, X_2), axis=1)
    print('X', X, X.shape)

    X_mpe = np.copy(X)
    X_mpe[np.arange(N) % 2 == 0, 0] = np.nan
    X_mpe[np.arange(N) % 2 == 1, 1] = np.nan
    print('X_mpe', X_mpe, X_mpe.shape)

    lls = likelihood(p, X)
    print('LLS', lls)

    mpe_ass, mpe_lls = mpe(p, X_mpe)
    print('MPE ', mpe_ass,  mpe_lls)

    ################################################################
    #
    # testing on a more involved SPN structure
    N = 10
    X_1 = rand_gen.normal(loc=0, scale=1, size=(N, 1))
    X_2 = rand_gen.poisson(lam=3, size=(N, 1))
    X_3 = rand_gen.exponential(scale=1 / 3, size=(N, 1))
    X = np.concatenate((X_1, X_2, X_3), axis=1)
    print('X', X, X.shape)
    X_mpe = np.copy(X)
    X_mpe[np.arange(N) % 3 == 0, 0] = np.nan
    X_mpe[np.arange(N) % 3 == 1, 1] = np.nan
    X_mpe[np.arange(N) % 3 == 2, 2] = np.nan
    print('X_mpe', X_mpe, X_mpe.shape)

    #
    # root is a sum
    root = Sum()

    #
    # two product nodes
    l_prod = Product()
    r_prod = Product()
    root.children = [l_prod, r_prod]
    root.weights = np.array([0.75, 0.25])

    #
    # right branch, three leaves
    right_leaf_1 = Gaussian(mean=0.5, stdev=2, scope=[0])
    right_leaf_2 = Poisson(mean=5, scope=[1])
    right_leaf_3 = Exponential(l=5, scope=[2])
    r_prod.children = [right_leaf_1, right_leaf_2, right_leaf_3]

    #
    # left branch one leaf and one sum nodes

    left_leaf_3 = Exponential(l=2, scope=[2])
    left_sum_node = Sum()
    l_prod.children = [left_sum_node, left_leaf_3]

    l_l_prod = Product()
    l_r_prod = Product()
    left_sum_node.children = [l_l_prod, l_r_prod]
    left_sum_node.weights = np.array([0.3, 0.7])

    #
    # far left branch, two leaves
    a_right_leaf_1 = Gaussian(mean=0, stdev=1, scope=[0])
    a_right_leaf_2 = Poisson(mean=10, scope=[1])
    l_l_prod.children = [a_right_leaf_1, a_right_leaf_2]

    #
    # far left branch, two leaves
    b_right_leaf_1 = Gaussian(mean=1, stdev=1, scope=[0])
    b_right_leaf_2 = Poisson(mean=11, scope=[1])
    l_r_prod.children = [b_right_leaf_1, b_right_leaf_2]

    #
    # composing
    rebuild_scopes_bottom_up(root)
    assign_ids(root)
    print(root)
    print(spn_to_str_equation(root))
    for n in get_nodes_by_type(root, Node):
        print(n, n.id)
        if isinstance(n, Sum):
            print(n.children, n.weights)

    lls = likelihood(root, X)
    print('LLS', lls)

    mpe_ass, mpe_lls = mpe(root, X_mpe)
    print('MPE ', mpe_ass,  mpe_lls)
