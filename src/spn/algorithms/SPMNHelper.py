"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""


import numpy as np
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Categorical

# below functions are used by learn_spmn_structure

def get_ds_context_prod(curr_train_data, scope, index, scope_index, params):
    """
    returns the Context object of spflow to use with split_cols, learn_mspn or learn_parametric methods of spflow while creating product node for spmn
    """
    n = curr_train_data.shape[1]
    scope_var = params.feature_names[scope_index:scope_index+n]
    context = []

    # if parametric, all variables are meta type -- categorical
    if params.util_to_bin:
        context = [Categorical]*n
        ds_context = Context(parametric_types=context, scope=scope, feature_names=scope_var).add_domains(curr_train_data)

    # if mixed, utilty is meta type -- real
    else:
        if params.utility_node[0] in scope_var:
            context = [MetaType.DISCRETE] * (n-1)
            context.append(MetaType.REAL)
        else:
            context = [MetaType.DISCRETE] * (n)

        scope = scope
        ds_context = Context(meta_types=context, scope=scope, feature_names=scope_var).add_domains(curr_train_data)
    return ds_context


def get_ds_context_sum(curr_train_data, scope, index, scope_index, params):
    """
    returns the Context object of spflow to use with split_rows method while creating sum node for spmn

    """
    n = curr_train_data.shape[1]
    curr_var_set_sum = params.partial_order[index: len(params.partial_order) + 1]
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

def get_curr_train_data_prod(train_dataa, curr_var_set):
    """
    :param train_dataa: all train_data from current index to end
    :param curr_var_set: current information set
    :return: split of train_data into two sets, one with current information set and the other is rest of the data
    """

    curr_var_set = curr_var_set
    train_data = train_dataa

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

def get_scope_prod(curr_train_data_prod, scope_index, scope_vars):
    """
    :return: returns scope of set of variables of current information set based on its index value in scope_variables
    """
    length = curr_train_data_prod.shape[1]
    assert scope_index + length <= len(scope_vars), "range of scope exceeds lenth of scope variables"
    scope = list(range(scope_index, scope_index + length))
    return scope





