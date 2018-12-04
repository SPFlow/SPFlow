"""
Created on November 23, 2018

@author: Alejandro Molina
"""
import sympy.stats as st
import sympy as sp
from spn.structure.leaves.parametric.Parametric import *

from spn.io.Symbolic import add_node_to_sympy


def get_density(dist, node, input_vars):
    return st.density(dist)(input_vars[node.scope[0]])


def gaussian_to_sympy(node, input_vars=None, log=False):
    result = get_density(st.Normal("Node%s" % node.id, node.mean, node.stdev), node, input_vars)
    if log:
        result = sp.log(result)
    return result


def gamma_to_sympy(node, input_vars=None, log=False):
    x = input_vars[node.scope[0]]
    scale = 1.0 / node.beta
    result = st.density(st.Gamma("Node%s" % node.id, node.alpha, 1), node, input_vars)(x / scale) / scale
    if log:
        result = sp.log(result)
    return result


def lognormal_to_sympy(node, input_vars=None, log=False):
    result = get_density(st.LogNormal("Node%s" % node.id, node.mean, node.stdev), node, input_vars)
    if log:
        result = sp.log(result)
    return result


def poisson_to_sympy(node, input_vars=None, log=False):
    result = get_density(st.Poisson("Node%s" % node.id, node.mean), node, input_vars)
    if log:
        result = sp.log(result)
    return result


def bernoulli_to_sympy(node, input_vars=None, log=False):
    result = get_density(st.Bernoulli("Node%s" % node.id, node.p), node, input_vars)
    if log:
        result = sp.log(result)
    return result


def categorical_to_sympy(node, input_vars=None, log=False):
    cat_param = {i: p for i, p in enumerate(node.p)}
    result = get_density(st.FiniteRV("Node%s" % node.id, cat_param), node, input_vars)
    if log:
        result = sp.log(result)
    return result


def geometric_to_sympy(node, input_vars=None, log=False):
    result = get_density(st.Geometric("Node%s" % node.id, node.p), node, input_vars)
    if log:
        result = sp.log(result)
    return result


def exponential_to_sympy(node, input_vars=None, log=False):
    result = get_density(st.Exponential("Node%s" % node.id, node.l), node, input_vars)
    if log:
        result = sp.log(result)
    return result


def uniform_to_sympy(node, input_vars=None, log=False):
    raise NotImplementedError()


def categorical_dictionary_to_sympy(node, input_vars=None, log=False):
    result = get_density(st.FiniteRV("Node%s" % node.id, node.p), node, input_vars)
    if log:
        result = sp.log(result)
    return result


def add_parametric_symbolic_support():
    add_node_to_sympy(Gaussian, gaussian_to_sympy)
    add_node_to_sympy(Gamma, gamma_to_sympy)
    add_node_to_sympy(LogNormal, lognormal_to_sympy)
    add_node_to_sympy(Poisson, poisson_to_sympy)
    add_node_to_sympy(Bernoulli, bernoulli_to_sympy)
    add_node_to_sympy(Categorical, categorical_to_sympy)
    add_node_to_sympy(Geometric, geometric_to_sympy)
    add_node_to_sympy(Exponential, exponential_to_sympy)
    add_node_to_sympy(Uniform, uniform_to_sympy)
    add_node_to_sympy(CategoricalDictionary, categorical_dictionary_to_sympy)
