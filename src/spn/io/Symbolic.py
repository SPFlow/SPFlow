"""
Created on November 23, 2018

@author: Alejandro Molina
"""
import operator
from functools import reduce
import sympy as sp

from spn.structure.Base import Sum, Product, eval_spn_bottom_up


def prod_to_sympy(node, children, input_vars=None, log=False):
    if log:
        return sum(children)
    result = reduce(operator.mul, children, 1)
    return result


def sum_to_sympy(node, children, input_vars=None, log=False):
    if not log:
        children_eval = map(lambda i: children[i] * node.weights[i], range(len(children)))

        return sum(children_eval)

    children_eval = map(lambda i: sp.exp(children[i]) * node.weights[i], range(len(children)))

    result = sum(children_eval)

    return sp.log(result)


_node_to_sympy = {Sum: sum_to_sympy, Product: prod_to_sympy}


def add_node_to_sympy(node_type, lambda_func):
    _node_to_sympy[node_type] = lambda_func


def spn_to_sympy(spn, node_to_sympy=_node_to_sympy, log=False):
    input_vars = sp.symbols("x:%s" % len(spn.scope))

    sympy_ecc = eval_spn_bottom_up(spn, node_to_sympy, input_vars=input_vars, log=log)

    return sympy_ecc
