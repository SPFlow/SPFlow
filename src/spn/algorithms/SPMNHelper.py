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
    ds_context = Context(
            meta_types=[params.meta_types[i] for i in scope],
            scope=scope,
            feature_names=scope_var
        )
    ds_context.add_domains(data)
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
        clustered_data_for_dec_val = data[data[:, 0] == dec_vals[i]]
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
    logging.debug(f'given scope of slice {slice_scope}')
    column_indices_of_slice_scope = [ind for ind, scope in enumerate(data_scope) if scope in slice_scope]
    logging.debug(f'column_indices_of_slice_scope are {column_indices_of_slice_scope}')

    data = data[:, column_indices_of_slice_scope]

    return data


def get_split_rows_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17):

    from spn.algorithms.splitting.Base import preproc, split_data_by_clusters

    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        from sklearn.cluster import KMeans
        km_model = KMeans(n_clusters=n_clusters, random_state=seed)
        clusters = km_model.fit_predict(data)
        return split_data_by_clusters(local_data, clusters, scope, rows=True), km_model

    return split_rows_KMeans


def get_row_indices_of_cluster(labels_array, cluster_num):
    return np.where(labels_array == cluster_num)[0]


def row_slice_data_by_indices(data, indices):
    return data[indices, :]
