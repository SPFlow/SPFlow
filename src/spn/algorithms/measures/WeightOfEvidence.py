"""
Created on Novenber 14, 2018

@author: Zhongjie Yu
@author: Alejandro Molina

"""
import numpy as np

from spn.algorithms.Inference import likelihood


def weight_of_evidence(spn, y_index, x_instance, n, k):
    """
    WE_i(Y|X)
    calculate the weight of evidence of every non-NaN index in x_instance, except for y
    :param spn:
    :param y_index:      index of y in vector x_instance
    :param x_instance:   vector, with value of requested RVs(and y) and NaN of non-requested RVs
    :param n:            number of training instances
    :param k:            cardinality of y
    :return:
    !!! vector x_instance includes the value at y th index
    """
    # P(Y|X)
    p_y_given_x = conditional_probability(spn, y_index, x_instance)
    # prepare output vector
    we = np.copy(x_instance)
    we[0][y_index] = np.nan
    # index of non-NaN RVs
    list_xi = np.argwhere(~np.isnan(we))
    # for each xi in vector x_without_y
    # get p_(y|x_without_y \ xi)
    for i, a in enumerate(list_xi):
        x_i = np.copy(x_instance)
        x_i[0][a[1]] = np.nan
        # P(Y|X\i) -> we
        p_y_given_x_i = conditional_probability(spn, y_index, x_i)
        we[0][a[1]] = def_w_of_e(laplace(p_y_given_x, n, k), laplace(p_y_given_x_i, n, k))

    return we


def laplace(p, n, k):
    assert 0 <= p <= 1, "Probability out of [0, 1]!"
    return (p * n + 1) / (n + k)


def def_w_of_e(p1, p2):
    """
    definition of weight of evidence, from https://arxiv.org/pdf/1702.04595.pdf
    :param p1:   P(class_label Y|instance X)
    :param p2:   P(class_label Y|instance X without Xi)
    :param n:    number of training instances
    :param k:    cardinality of y
    :return:
    """
    w_of_e = np.log2(odds(p1)) - np.log2(odds(p2))
    return w_of_e


def odds(p):
    """
    definition of odds ratio, without Laplace correction
    :param p:    probability
    :param n:    number of training instances
    :param k:    cardinality of y
    :return:     odds ratio
    """
    assert 0 <= p <= 1, "Probability out of [0, 1]!"
    y = p / (1 - p)
    return y


def conditional_probability(spn, y_index, x_instance):
    """
    calculation of conditional probability P(Y|X)
    :param spn:
    :param y_index:      index of y in vector x_instance
    :param x_instance:   vector, with value of requested RVs(and y) and NaN of non-requested RVs
    :return:
    vector x_instance includes the value at y th index
    """
    x_without_y = np.copy(x_instance)
    x_without_y[0][y_index] = np.nan
    # P(Y|X)
    p_y_given_x = likelihood(spn, x_instance) / likelihood(spn, x_without_y)

    return p_y_given_x
