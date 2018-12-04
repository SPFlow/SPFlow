"""
Created on Novenber 14, 2018

@author: Zhongjie Yu
@author: Alejandro Molina

"""
import numpy as np


def conditional_mutual_information(spn, ds_context, X, Y, cond_Z, debug=False):
    """
    calculate the conditional mutual information
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
    :param spn:          SPN
    :param ds_context:
    :param X:            set of scope integers representing RVs X
    :param Y:            set of scope integers representing RVs Y
    :param cond_Z:       set of scope integers representing RVs which are given
    :return:             conditional mutual information
    """
    # first check if the RVs are all DISCRETE
    check_discrete(ds_context, X, Y, cond_Z)
    # print the range of each RV for debug, works when X and Y are scalars
    if debug:
        print_debug_info(X, Y, cond_Z)
    # print progress if debug
    hz = entropy(spn, ds_context, cond_Z, debug)
    hxz = entropy(spn, ds_context, X | cond_Z, debug)
    hyz = entropy(spn, ds_context, Y | cond_Z, debug)
    hxyz = entropy(spn, ds_context, X | Y | cond_Z, debug)
    # by definition
    cmi = hxz + hyz - hz - hxyz
    if debug:
        print("I(%s,%s|%s)=%s" % (X, Y, cond_Z, cmi))
    return cmi


def mutual_information(spn, ds_context, Xset, Yset, debug=False):
    """
    I(X,Y) = H(X) + H(Y) - H(X,Y)
    :param spn:
    :param ds_context:
    :param Xset:        set of scope integers representing RVs X
    :param Yset:        set of scope integers representing RVs Y
    :return:
    """
    # by definition, MI(x,y) = h(x) + h(y) - h(xy)
    # where h is the entropy
    hx = entropy(spn, ds_context, Xset, debug)
    hy = entropy(spn, ds_context, Yset, debug)
    hxy = entropy(spn, ds_context, Xset | Yset, debug)
    mi = hx + hy - hxy

    if debug:
        print("I(%s)=%s" % (Xset | Yset, mi))

    return mi


def entropy(spn, ds_context, RVset, debug=False):
    """
    calc the entropy from spn and the permutation of RVs
    :param spn:       input SPN
    :param ds_context:
    :param RVset:     set of scope integers representing RVs
    :return:          entropy of RVs
    """
    # get permutation of RVset
    perm_RV = get_permutation(ds_context, RVset)
    # get entropy
    from spn.algorithms.Inference import log_likelihood

    log_p = log_likelihood(spn, perm_RV)
    log_p[np.isinf(log_p)] = 0
    h = np.exp(log_p) * log_p
    # check, if p==0, log_p will be "-np.inf" and h will be NaN
    # if h==NaN, setting it 0 makes entropy=0
    h[np.isnan(h)] = 0
    H = -(h.sum())

    if debug:
        print("H(%s)=%s" % (RVset, H))

    return H


def get_permutation(ds_context, RVset):
    """
    get the permutation of given RVset of discrete RVs
    :param ds_context:   context of RVs
    :param RVset:        set of scope integers representing RVs
    :return:             full permutation of RVs in the form of a matrix
    """

    def cartesian_product(arrays):
        # https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    perm = cartesian_product(list(map(lambda rv: np.array(ds_context.domains[rv]), RVset)))

    result = np.empty((perm.shape[0], ds_context.domains.shape[0]))
    result[:] = np.nan

    for i, rv in enumerate(RVset):
        result[:, rv] = perm[:, i]

    return result


def print_debug_info(ds_context, X, Y, cond_Z):
    """
    print |X| * |Y| * |Z| in debug to have an overview of complexity
    :param ds_context:
    :param X:
    :param Y:
    :param cond_Z:
    :return:
    !!!
    important: when computing the (conditional) mutual info with large
               number of RVs, the calculation is slow. Setting debug=TRUE
               shows the entropy which is on calculation.
    !!!
    """
    v4print = " "
    c_Z = np.array(list(cond_Z))
    x = np.array(list(X))
    y = np.array(list(Y))
    for index in c_Z:
        v4print = v4print + str(ds_context.domains[index].size) + " * "
    print(
        "Number of permutation in CMI: n1(",
        ds_context.domains[x[0]].size,
        "), n2(",
        ds_context.domains[y[0]].size,
        "). { ",
        v4print[:-2],
        "}.",
    )


def check_discrete(ds_context, X, Y, cond_Z):
    # make sure the variables are all discrete RVs
    u_list = list(X | Y | cond_Z)
    for i in u_list:
        assert (
            ds_context.meta_types[i].name == "DISCRETE"
        ), "The function of (Conditional) Mutual Information supports DISCRETE random variables only!"
