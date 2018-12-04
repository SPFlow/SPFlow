"""
Created on Ocotber 27, 2018

@author: Nicola Di Mauro
"""

import logging
from collections import deque
from spn.algorithms.StructureLearning import Operation, default_slicer

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np

from spn.algorithms.TransformStructure import Prune
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, assign_ids
import multiprocessing
import os

cpus = max(1, os.cpu_count() - 2)  # - int(os.getloadavg()[2])
pool = multiprocessing.Pool(processes=cpus)


def get_next_operation_cnet(min_instances_slice=100, min_features_slice=1):
    def next_operation_cnet(data, scope):

        minimalFeatures = len(scope) == min_features_slice
        minimalInstances = data.shape[0] <= min_instances_slice

        if minimalFeatures or minimalInstances:
            return Operation.CREATE_LEAF, None
        else:
            return Operation.CONDITIONING, None

    return next_operation_cnet


def learn_structure_cnet(
    dataset,
    ds_context,
    conditioning,
    create_leaf,
    next_operation_cnet=get_next_operation_cnet(),
    initial_scope=None,
    data_slicer=default_slicer,
):
    assert dataset is not None
    assert ds_context is not None
    assert create_leaf is not None
    assert next_operation_cnet is not None

    root = Product()
    root.children.append(None)

    if initial_scope is None:
        initial_scope = list(range(dataset.shape[1]))

    tasks = deque()
    tasks.append((dataset, root, 0, initial_scope))

    while tasks:

        local_data, parent, children_pos, scope = tasks.popleft()

        operation, op_params = next_operation_cnet(local_data, scope)

        logging.debug("OP: {} on slice {} (remaining tasks {})".format(operation, local_data.shape, len(tasks)))

        if operation == Operation.CONDITIONING:
            from spn.algorithms.splitting.Base import split_data_by_clusters

            conditioning_start_t = perf_counter()

            col_conditioning, found_conditioning = conditioning(local_data)

            if not found_conditioning:
                node = create_leaf(local_data, ds_context, scope)
                parent.children[children_pos] = node

                continue

            clusters = (local_data[:, col_conditioning] == 1).astype(int)
            data_slices = split_data_by_clusters(local_data, clusters, scope, rows=True)

            node = Sum()
            node.scope.extend(scope)
            parent.children[children_pos] = node

            for data_slice, scope_slice, proportion in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"

                node.weights.append(proportion)

                product_node = Product()
                node.children.append(product_node)
                node.children[-1].scope.extend(scope)

                right_data_slice = np.hstack(
                    (data_slice[:, :col_conditioning], data_slice[:, (col_conditioning + 1) :])
                ).reshape(data_slice.shape[0], data_slice.shape[1] - 1)
                product_node.children.append(None)
                tasks.append(
                    (
                        right_data_slice,
                        product_node,
                        len(product_node.children) - 1,
                        scope_slice[:col_conditioning] + scope_slice[col_conditioning + 1 :],
                    )
                )

                left_data_slice = data_slice[:, col_conditioning].reshape(data_slice.shape[0], 1)
                product_node.children.append(None)
                tasks.append(
                    (left_data_slice, product_node, len(product_node.children) - 1, [scope_slice[col_conditioning]])
                )

            conditioning_end_t = perf_counter()
            logging.debug("\t\tconditioning  (in {:.5f} secs)".format(conditioning_end_t - conditioning_start_t))

            continue

        elif operation == Operation.CREATE_LEAF:
            cltree_start_t = perf_counter()
            node = create_leaf(local_data, ds_context, scope)
            parent.children[children_pos] = node
            cltree_end_t = perf_counter()
        else:
            raise Exception("Invalid operation: " + operation)

    node = root.children[0]
    assign_ids(node)
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err
    node = Prune(node)
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err

    return node
