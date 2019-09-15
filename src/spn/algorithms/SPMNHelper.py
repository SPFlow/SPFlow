"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""

import numpy as np
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Categorical
import logging


# below functions are used by learn_spmn_structure
def get_ds_context(data, scope, params):
    """
    :param data: numpy array of data for Context object
    :param scope: scope of data
    :param params: params of SPMN
    :return: Context object of SPFlow
    """

    num_of_variables = data.shape[1]
    scope_var = np.array(params.feature_names)[scope].tolist()
    # if parametric, all variables are type -- categorical
    if params.util_to_bin:
        context = [Categorical] * num_of_variables
        ds_context = Context(parametric_types=context, scope=scope, feature_names=scope_var).add_domains(data)

    # if mixed, utility is meta type -- UTILITY
    else:
        if params.utility_node[0] in scope_var:
            context = [MetaType.DISCRETE] * (num_of_variables - 1)
            context.append(MetaType.UTILITY)
        else:
            context = [MetaType.DISCRETE] * num_of_variables

        scope = scope
        ds_context = Context(meta_types=context, scope=scope, feature_names=scope_var).add_domains(data)
    return ds_context


def cluster(data, dec_vals):
    """
    :param data: numpy array of data containing variable at 0th column on whose values cluster is needed
    :param dec_vals: values of variable at that 0th column
    :return: clusters of data (excluding the variable at 0th column) grouped together based on values of the variable
    """

    logging.debug(f'in cluster function of SPMNHelper')
    clusters_on_remaining_columns = []
    for i in range(0, len(dec_vals)):
        clustered_data_for_dec_val = data[[data[:, 0] == dec_vals[i]]]
        # exclude the 0th column, which belongs to decision node
        clustered_data_on_remaining_columns = np.delete(clustered_data_for_dec_val, 0, 1)
        # logging.debug(f'clustered data on remaining columns is {clustered_data_on_remaining_columns}')

        clusters_on_remaining_columns.append(clustered_data_on_remaining_columns)

    logging.debug(f'{len(clusters_on_remaining_columns)} clusters formed on remaining columns based on decision values')
    return clusters_on_remaining_columns


def split_on_decision_node(data):
    """
    :param data: numpy array of data with decision node at 0th column
    :return: clusters split on values of decision node
    """

    logging.debug(f'in split_on_decision_node function of SPMNHelper')
    # logging.debug(f'data at decision node is {data}')
    dec_vals = np.unique(data[:, 0])   # since 0th column of current train data is decision node
    logging.debug(f'dec_vals are {dec_vals}')
    # cluster remaining data based on decision values
    clusters_on_remaining_columns = cluster(data, dec_vals)
    return clusters_on_remaining_columns, dec_vals


def column_slice_data_by_scope(data, data_scope, slice_scope):
    """
    :param data:  numpy array of data, columns ordered in data_scope order
    :param data_scope: scope of variables of the given data
    :param slice_scope: scope of the variables whose data slice is required
    :return: numpy array of data that corresponds to the variables of the given scope
    """

    # assumption, data columns are ordered in data_scope order
    logging.debug(f'in column_slice_data_by_scope function of SPMNHelper')
    data_indices_of_slice_scope = [ind for ind, scope in enumerate(data_scope) if scope in slice_scope]
    logging.debug(f'data_indices_of_slice_scope are {data_indices_of_slice_scope}')

    data = data[:, data_indices_of_slice_scope]

    return data




