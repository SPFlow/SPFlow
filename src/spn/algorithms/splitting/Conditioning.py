"""
Created on October 26, 2018

@author: Nicola Di Mauro
"""

import numpy as np

from spn.algorithms.splitting.Base import split_data_by_clusters


def get_split_rows_random_conditioning():
    def split_rows_random_conditioning(local_data):
        choice = np.random.choice(local_data.shape[1], local_data.shape[1], replace=False)
        for col_conditioning in choice:
            ones = np.sum(local_data[:, col_conditioning])
            if ones > 0 or ones < local_data.shape[0]:
                return col_conditioning, True
        return None, True

    return split_rows_random_conditioning


def naive_ll(local_data, alpha=0.1):
    ones = np.sum(local_data, axis=0)
    zeros = local_data.shape[0] - ones
    probs = (ones + alpha) / (local_data.shape[0] + 2 * alpha)
    o_log_probs = np.log(probs)
    z_log_probs = np.log(1 - probs)
    ll = (np.sum(ones * o_log_probs) + np.sum(zeros * z_log_probs)) / local_data.shape[0]
    return ll


def get_split_rows_naive_mle_conditioning():
    def split_rows_naive_mle_conditioning(local_data):
        # mle conditioning

        original_ll = naive_ll(local_data)

        scope = [i for i in range(local_data.shape[1])]

        best_col_conditioning = None
        best_conditioning_ll = -np.inf
        for col_conditioning in range(local_data.shape[1]):
            ones = np.sum(local_data[:, col_conditioning])
            if ones == 0 or ones == local_data.shape[0]:
                continue

            clusters = (local_data[:, col_conditioning] == 1).astype(int)
            data_slices = split_data_by_clusters(local_data, clusters, scope, rows=True)

            left_data_slice, left_scope_slice, left_proportion = data_slices[0]
            right_data_slice, right_scope_slice, right_proportion = data_slices[1]

            left_data_slice = np.hstack(
                (left_data_slice[:, :col_conditioning], left_data_slice[:, (col_conditioning + 1) :])
            ).reshape(left_data_slice.shape[0], left_data_slice.shape[1] - 1)
            right_data_slice = np.hstack(
                (right_data_slice[:, :col_conditioning], right_data_slice[:, (col_conditioning + 1) :])
            ).reshape(right_data_slice.shape[0], right_data_slice.shape[1] - 1)

            left_ll = naive_ll(left_data_slice)
            right_ll = naive_ll(right_data_slice)

            conditioning_ll = (
                (left_ll + np.log(left_proportion)) * left_data_slice.shape[0]
                + (right_ll + np.log(right_proportion)) * right_data_slice.shape[0]
            ) / local_data.shape[0]
            if conditioning_ll > best_conditioning_ll:
                best_conditioning_ll = conditioning_ll
                best_col_conditioning = col_conditioning

        return best_col_conditioning, best_col_conditioning != None

    return split_rows_naive_mle_conditioning
