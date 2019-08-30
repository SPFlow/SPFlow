"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""


import numpy as np
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Categorical

# below functions are used by learn_spmn_structure

def get_ds_context_prod(curr_train_data, scope, params):
    """
    returns the Context object of spflow to use with split_cols, learn_mspn or learn_parametric methods of spflow while creating product node for spmn
    """
    n = curr_train_data.shape[1]
    scope_var = np.array(params.feature_names)[scope].tolist()
    context = []

    # if parametric, all variables are type -- categorical
    if params.util_to_bin:
        context = [Categorical]*n
        ds_context = Context(parametric_types=context, scope=scope, feature_names=scope_var).add_domains(curr_train_data)

    # if mixed, utilty is meta type -- real
    else:
        if params.utility_node[0] in scope_var:
            context = [MetaType.DISCRETE] * (n-1)
            context.append(MetaType.UTILITY)
        else:
            context = [MetaType.DISCRETE] * (n)

        scope = scope
        ds_context = Context(meta_types=context, scope=scope, feature_names=scope_var).add_domains(curr_train_data)
    return ds_context


def get_ds_context_sum(curr_train_data, scope, index, params):
    """
    returns the Context object of spflow to use with split_rows method while creating sum node for spmn

    """
    n = curr_train_data.shape[1]
    curr_var_set_sum = params.partial_order[index: len(params.partial_order) + 1]
    del curr_var_set_sum[0]
    curr_var_set_sum = [scope] + curr_var_set_sum
    curr_var_set_sum1 = [var for curr_var_set in curr_var_set_sum for var in curr_var_set]

    if params.util_to_bin:
        context = [Categorical]*n
        ds_context = Context(parametric_types=context, scope=scope, feature_names=curr_var_set_sum1).add_domains(curr_train_data)

    # utilty is meta type -- real
    else:

        if params.utility_node[0] in curr_var_set_sum1:
            context = [MetaType.DISCRETE] * (n-1)
            context.append(MetaType.REAL)
        else:
            context = [MetaType.DISCRETE] * (n)
        scope = scope
        ds_context = Context(meta_types=context, scope=scope, feature_names=curr_var_set_sum1).add_domains(curr_train_data)

    return ds_context

def cluster(train_data, dec_vals):
    """

    :param dec_vals: values of variable
    :return: clusters of train_data grouped together based on values of the variables
    """
    cl=[]
    for i in range(0, len(dec_vals)):
        train_data1 = train_data[[train_data[:, 0] == dec_vals[i]]]
        train_data2 = np.delete(train_data1, 0, 1)
        cl.append(train_data2)

    return cl

def split_on_decision_node(train_data, decision_node=None) :
    """

    :param train_data: current train data with decision node at 0th column
    :param decision_node:
    :return: clusters split on values of decision node
    """

    train_data = train_data
    dec_vals = np.unique(train_data[:, 0])   #since 0th column of current train data is decision node
    cl = cluster(train_data, dec_vals)
    return cl, dec_vals

def get_curr_train_data_prod(train_data, curr_var_set):
    """
    :param train_data: all train_data from current index to end
    :param curr_var_set: current information set
    :return: split of train_data into two sets, one with current information set and the other is rest of the data
    """

    slice = len(curr_var_set)
    curr_train_data_prod = train_data[:, :slice]
    rest_train_data = train_data[:, slice:]
    #print(curr_train_data_prod)
    return curr_train_data_prod, rest_train_data


def set_next_operation(op="None"):
    global next_op
    next_op = op

def get_next_operation():
     return next_op

def get_scope_prod(curr_scope, independent_vars=None):
    """
    :param curr_scope: np array of indices of variables in current scope
    :param independent_vars: boolean array of whether each variable has passed independence testing
    :return: returns scope of set of variables of current information set based on its index value in scope_variables
    """
    if independent_vars:
        scope_prod = curr_scope[independent_vars].tolist()
        indices = np.array(range(curr_scope.shape[0]))[independent_vars]
        scope_rest = np.delete(curr_scope, indices).tolist()
    else:
        scope_prod = curr_scope.tolist()
        scope_rest = None
    return scope_prod, scope_rest
