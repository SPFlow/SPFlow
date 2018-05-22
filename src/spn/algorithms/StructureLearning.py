'''
Created on March 20, 2018

@author: Alejandro Molina
'''
import logging
from collections import deque
from enum import Enum
try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


import numpy as np

from spn.algorithms.Pruning import prune
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, assign_ids


class Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLUMNS = 2
    SPLIT_ROWS = 3
    NAIVE_FACTORIZATION = 4
    REMOVE_UNINFORMATIVE_FEATURES = 5


def get_next_operation(min_instances_slice=100):
    def next_operation(data, no_clusters=False, no_independencies=False, is_first=False, cluster_first=True, cluster_univariate=False):

        minimalFeatures = data.shape[1] == 1
        minimalInstances = data.shape[0] <= min_instances_slice

        if minimalFeatures:
            if minimalInstances or no_clusters:
                return Operation.CREATE_LEAF
            else:
                if cluster_univariate:
                    return Operation.SPLIT_ROWS
                else:
                    return Operation.CREATE_LEAF

        ncols_zero_variance = np.sum(np.var(data, 0) == 0)
        if ncols_zero_variance > 0:
            if ncols_zero_variance == data.shape[1]:
                return Operation.NAIVE_FACTORIZATION
            else:
                return Operation.REMOVE_UNINFORMATIVE_FEATURES

        if minimalInstances or (no_clusters and no_independencies):
            return Operation.NAIVE_FACTORIZATION

        if no_independencies:
            return Operation.SPLIT_ROWS

        if no_clusters:
            return Operation.SPLIT_COLUMNS

        if is_first:
            return Operation.SPLIT_ROWS if cluster_first else Operation.SPLIT_COLUMNS

        return Operation.SPLIT_COLUMNS

    return next_operation


def learn_structure(dataset, ds_context, split_rows, split_cols, create_leaf, next_operation=get_next_operation()):
    assert dataset is not None
    assert ds_context is not None
    assert split_rows is not None
    assert split_cols is not None
    assert create_leaf is not None
    assert next_operation is not None

    root = Product()
    root.children.append(None)

    tasks = deque()
    tasks.append((dataset, root, 0, list(range(dataset.shape[1])), False, False))

    while tasks:

        local_data, parent, children_pos, scope, no_clusters, no_independencies = tasks.popleft()

        operation = next_operation(local_data, no_clusters=no_clusters, no_independencies=no_independencies,
                                   is_first=(parent is root))

        logging.debug('OP: {} on slice {} (remaining tasks {})'.format(operation,
                                                                      local_data.shape,
                                                                      len(tasks)))

        if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:
            variances = np.var(local_data, axis=0)

            node = Product()
            node.scope.extend(scope)
            parent.children[children_pos] = node

            cols = []
            for col in range(local_data.shape[1]):
                if variances[col] == 0:
                    node.children.append(None)
                    tasks.append((local_data[:, col].reshape((-1, 1)), node,
                                  len(node.children) - 1, [scope[col]], True, True))
                else:
                    cols.append(col)

            node.children.append(None)

            c_pos = len(node.children) - 1
            if len(cols) == 1:
                col = cols[0]

                tasks.append((local_data[:, col].reshape((-1, 1)),
                              node, c_pos, [scope[col]], True, True))
            else:
                tasks.append((local_data[:, cols], node, c_pos,
                              np.array(scope)[cols].tolist(), False, False))

            continue

        elif operation == Operation.SPLIT_ROWS:

            split_start_t = perf_counter()
            data_slices = split_rows(local_data, ds_context, scope)
            split_end_t = perf_counter()
            logging.debug('\t\tfound {} row clusters (in {:.5f} secs)'.format(len(data_slices),
                                                                             split_end_t - split_start_t))

            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, True, False))
                continue

            node = Sum()
            node.scope.extend(scope)
            parent.children[children_pos] = node

            for data_slice, scope_slice in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"

                node.children.append(None)
                node.weights.append(data_slice.shape[0] / local_data.shape[0])
                tasks.append((data_slice, node, len(node.children) - 1, scope, False, False))

            continue

        elif operation == Operation.SPLIT_COLUMNS:

            split_start_t = perf_counter()
            data_slices = split_cols(local_data, ds_context, scope)
            split_end_t = perf_counter()
            logging.debug('\t\tfound {} col clusters (in {:.5f} secs)'.format(len(data_slices),
                                                                             split_end_t - split_start_t))

            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, False, True))
                continue

            node = Product()
            node.scope.extend(scope)
            parent.children[children_pos] = node

            for data_slice, scope_slice in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"

                node.children.append(None)
                tasks.append((data_slice, node, len(node.children) - 1, scope_slice, False, False))
            continue

        elif operation == Operation.NAIVE_FACTORIZATION:
            node = Product()
            node.scope.extend(scope)
            parent.children[children_pos] = node

            split_start_t = perf_counter()
            for col in range(local_data.shape[1]):
                node.children.append(None)
                tasks.append((local_data[:, col].reshape((-1, 1)), node,
                              len(node.children) - 1, [scope[col]], True, True))
            split_end_t = perf_counter()

            logging.debug('\t\tsplit {} columns (in {:.5f} secs)'.format(local_data.shape[1],
                                                                        split_end_t - split_start_t))

            continue

        elif operation == Operation.CREATE_LEAF:

            leaf_start_t = perf_counter()
            node = create_leaf(local_data, ds_context, scope)
            # node.scope.extend(scope)
            parent.children[children_pos] = node
            leaf_end_t = perf_counter()

            logging.debug('\t\t created leaf {} for scope={} (in {:.5f} secs)'.format(node.__class__.__name__,
                                                                                     scope,
                                                                                     leaf_end_t - leaf_start_t))

        else:
            raise Exception('Invalid operation: ' + operation)

    node = root.children[0]

    assert is_valid(node), "invalid before pruning"
    node = prune(node)
    assert is_valid(node), "invalid after pruning"
    assign_ids(node)
    return node
